import torch
import torch.optim as optim
import torch.nn as nn
import os
from model.seq2seq import Seq2Seq
from model.encoder_decoder import Encoder, Decoder
from dataloader import get_dataloader

# Set device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")
print("Using device:", device)

# Hyperparameters
INPUT_DIM = 8000
OUTPUT_DIM = 8000
EMBED_SIZE = 256
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.5
LEARNING_RATE = 0.00005
EPOCHS = 10
BATCH_SIZE = 8
PAD_IDX = 0  

# Dataset Path
DATA_PATH = os.path.join(os.path.dirname(__file__), "../../dataset/processed_data.h5")

# Initialize Model
encoder = Encoder(INPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, device).to(device)
print(encoder)
decoder = Decoder(OUTPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, device).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

# Optimizer & Loss Function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Load Dataset
dataloader = get_dataloader(DATA_PATH, batch_size=BATCH_SIZE)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg)

        # **Fix: Ensure output requires gradient**
        #output.requires_grad = True

        # **Fix: Shape output & target correctly**
        output = output[1:].reshape(-1, OUTPUT_DIM)
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clipping to prevent NaN
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss / len(dataloader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "saved_baseline_model.pth")
print("Model training complete! Saved as 'saved_baseline_model.pth'.")
