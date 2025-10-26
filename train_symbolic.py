# train_symbolic.py
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from symbolic_controller import SymbolicLLM

def kl_on_anchor(logits_adapt, logits_base, mask):
    # symmetric KL on anchor prompts where behavior should not change
    p = F.log_softmax(logits_adapt, dim=-1)
    q = F.log_softmax(logits_base, dim=-1)
    kl = F.kl_div(p, q.exp(), reduction="none") + F.kl_div(q, p.exp(), reduction="none")
    kl = kl.sum(-1)
    return (kl * mask).sum() / (mask.sum() + 1e-8)

def lora_reg(controller):
    reg = 0.0
    for mod in controller.heads.values():
        reg += mod["A"].weight.pow(2).sum() + mod["B"].weight.pow(2).sum()
    return reg

def gate_l1(controller):
    gsum = 0.0
    for mod in controller.heads.values():
        # take the bias of "g" linear if present or compute average g over a dummy batch
        pass
    return 0.0  # keep simple; or measure g during forward

def train(model, train_ds, anchor_ds, epochs=2, lr=2e-4, lam=1e-6, rho=1e-4, gamma=1e-2, device="cuda"):
    model.to(device)
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    anchor_loader = DataLoader(anchor_ds, batch_size=2, shuffle=True)
    anchor_iter = iter(anchor_loader)

    for ep in range(epochs):
        for batch in train_loader:
            model.train()
            ctx_ids, ctx_mask, x_ids, x_mask, labels = [t.to(device) for t in batch]
            loss_adapt, logits_adapt = model.forward_with_context(ctx_ids, ctx_mask, x_ids, x_mask, labels)

            # stability on anchors
            try:
                abatch = next(anchor_iter)
            except StopIteration:
                anchor_iter = iter(anchor_loader)
                abatch = next(anchor_iter)
            actx_ids, actx_mask, ax_ids, ax_mask, alabels = [t.to(device) for t in abatch]
            with torch.no_grad():
                # base logits (no adapters)
                base_out = model.lm(input_ids=ax_ids, attention_mask=ax_mask)
                logits_base = base_out.logits
            # adapted logits
            _, logits_anchor_adapt = model.forward_with_context(actx_ids, actx_mask, ax_ids, ax_mask, alabels)
            loss_kl = kl_on_anchor(logits_anchor_adapt, logits_base, ax_mask)

            # regularizers
            loss_reg = lora_reg(model.controller)

            loss = loss_adapt + lam * loss_reg + gamma * loss_kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
