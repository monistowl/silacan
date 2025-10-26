# symbolic_controller.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- Config ----
BASE_MODEL = "meta-llama/Llama-3-8b"  # example; use small for local tests
TARGET_MODULES = ["q_proj", "v_proj"]  # LoRA on attention (per layer)
RANK = 8
ALPHA = 16.0
CTX_LAYER = -1  # which hidden states to pool for z
HIDDEN = 1024   # controller hidden size

# ---- Utility: find target linear modules ----
def iter_target_linears(model, target_names):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_names):
            yield name, module

# ---- Controller (Symbolic) ----
class SymbolicController(nn.Module):
    def __init__(self, d_model, rank=RANK, hidden=HIDDEN, n_layers=0, per_layer_dims=None):
        super().__init__()
        self.rank = rank
        self.d_model = d_model
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Per-target heads: for each W_l (dxd), emit A(dxr), B(dxr), gate in (0,1)
        self.heads = nn.ModuleDict()
        for lname, d in per_layer_dims.items():  # d = module.out_features (assuming square-ish)
            self.heads[lname] = nn.ModuleDict({
                "A": nn.Linear(hidden, d * rank),
                "B": nn.Linear(hidden, d * rank),
                "g": nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
            })

    def forward(self, z):
        h = self.trunk(z)  # [B, hidden]
        out = {}
        for lname, head in self.heads.items():
            A = head["A"](h)  # [B, d*r]
            B = head["B"](h)
            g = head["g"](h)  # [B, 1]
            out[lname] = {"A": A, "B": B, "g": g}
        return out

# ---- Adapter wrapper that applies predicted LoRA per batch ----
class HyperLoRAAdapter:
    def __init__(self, model, target_modules, rank=RANK, alpha=ALPHA):
        self.model = model
        self.rank = rank
        self.alpha = alpha
        # cache shapes and hooks
        self.targets = {}
        for lname, lin in iter_target_linears(model, target_modules):
            d_out, d_in = lin.weight.shape
            self.targets[lname] = {"module": lin, "shape": (d_out, d_in)}
        self._orig_weights = {}  # to restore after each forward

    def apply(self, hyper_out):
        # hyper_out[lname]: {"A": [B,d*r], "B":[B,d*r], "g":[B,1]}
        # We assume batch size = 1 for generation. For training, we can loop microbatches.
        for lname, pack in hyper_out.items():
            lin = self.targets[lname]["module"]
            d_out, d_in = self.targets[lname]["shape"]
            A = pack["A"].view(d_out, self.rank)
            B = pack["B"].view(d_out, self.rank)
            g = pack["g"].view(1).item()
            delta = g * (self.alpha / self.rank) * (A @ B.T)  # d_out x d_out (for square W)
            # if W is not square, adjust: A: d_out x r, B: d_in x r
            if delta.shape != lin.weight.data.shape:
                # fallback for non-square linear layers:
                d_out, d_in = lin.weight.data.shape
                A = pack["A"].view(d_out, self.rank)
                B = pack["B"].view(d_in, self.rank)
                delta = g * (self.alpha / self.rank) * (A @ B.T)  # d_out x d_in

            if lin not in self._orig_weights:
                self._orig_weights[lin] = lin.weight.data.clone()
            lin.weight.data = self._orig_weights[lin] + delta

    def restore(self):
        for lin, W in self._orig_weights.items():
            lin.weight.data = W
        self._orig_weights.clear()

# ---- End-to-end wrapper ----
class SymbolicLLM(nn.Module):
    def __init__(self, base_model_name=BASE_MODEL, target_modules=TARGET_MODULES, rank=RANK, alpha=ALPHA):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        self.lm = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
        for p in self.lm.parameters():
            p.requires_grad_(False)  # freeze base
        # enumerate target dims
        per_layer_dims = {}
        for lname, lin in iter_target_linears(self.lm, target_modules):
            per_layer_dims[lname] = lin.weight.data.shape[0]  # assume square-ish
        d_model = self.lm.config.hidden_size
        self.controller = SymbolicController(d_model, rank=rank, per_layer_dims=per_layer_dims)
        self.adapter = HyperLoRAAdapter(self.lm, target_modules, rank=rank, alpha=alpha)

    def pool_context(self, ctx_ids, attention_mask):
        # get hidden states at specified layer for context tokens and mean-pool
        outputs = self.lm.model(input_ids=ctx_ids, attention_mask=attention_mask, output_hidden_states=True)
        H = outputs.hidden_states[CTX_LAYER]  # [B, T, d]
        z = (H * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        return z  # [B,d]

    def forward_with_context(self, ctx_ids, ctx_mask, x_ids, x_mask, labels):
        # 1) make z
        z = self.pool_context(ctx_ids, ctx_mask)  # [B,d]
        # 2) emit LoRA deltas
        hyper_out = self.controller(z)  # dict of A,B,g per target
        # 3) apply deltas
        self.adapter.apply(hyper_out)
        # 4) usual LM forward
        out = self.lm(input_ids=x_ids, attention_mask=x_mask, labels=labels)
        loss = out.loss
        # 5) restore base weights
        self.adapter.restore()
        return loss, out.logits

    @torch.no_grad()
    def generate_with_context(self, ctx, prompt, max_new_tokens=64, temperature=0.7):
        device = next(self.parameters()).device
        ctx_ids = self.tokenizer(ctx, return_tensors="pt").to(device)
        z = self.pool_context(ctx_ids["input_ids"], ctx_ids["attention_mask"])
        hyper_out = self.controller(z)
        self.adapter.apply(hyper_out)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        out = self.lm.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
        self.adapter.restore()
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
