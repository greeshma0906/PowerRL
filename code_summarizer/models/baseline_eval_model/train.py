import torch
import torch.optim as optim
import torch.nn as nn
from model import Seq2Seq
from data_loader import get_dataloader

# Model Parameters
input_dim = 5000
output_dim = 5000
embed_size = 256
hidden_size = 512
num_layers = 2
dropout = 0.5

model = Seq2Seq(input_dim, output_dim, embed_size, hidden_size, num_layers, dropout)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


source_texts = ["def add(a, b): return a + b"]
target_texts = ["adds two numbers"]
source_vocab = {"<pad>": 0, "<unk>": 1, "def": 2, "return": 3}
target_vocab = {"<pad>": 0, "<unk>": 1, "adds": 2}

dataloader = get_dataloader(source_texts, target_texts, source_vocab, target_vocab)

for epoch in range(10):
    for src, trg in dataloader:
        optimizer.zero_grad()
        output = model(src, trg)
        loss = criterion(output.view(-1, output_dim), trg.view(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

torch.save(model.state_dict(), "saved_baseline_model.pth")
