#example use
from symbolic_controller import SymbolicLLM
import torch

model = SymbolicLLM()
model.eval().to("cuda")

ctx = "<ROLE=judge><NORM=formal><PRON=singular>"
prompt = "The defendant stated that"
print(model.generate_with_context(ctx, prompt))

ctx2 = "<ROLE=friend><NORM=casual><PRON=they>"
print(model.generate_with_context(ctx2, prompt))
