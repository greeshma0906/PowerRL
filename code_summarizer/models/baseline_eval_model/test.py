import torch
from model import Seq2Seq
from data_loader import get_dataloader

model = Seq2Seq(5000, 5000, 256, 512, 2, 0.5)
model.load_state_dict(torch.load("saved_model.pth"))
model.eval()

dataloader = get_dataloader(["def multiply(a, b): return a * b"], ["multiplies two numbers"], {"<pad>": 0, "<unk>": 1}, {"<pad>": 0, "<unk>": 1})

with torch.no_grad():
    for src, trg in dataloader:
        output = model(src, trg, 0)
        print(output.argmax(2))
