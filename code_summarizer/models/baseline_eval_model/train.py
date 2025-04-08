import torch
import torch.optim as optim
import torch.nn as nn
import os
import time
from model.seq2seq import Seq2Seq
from model.encoder_decoder import Encoder, Decoder
from dataloader import get_dataloader

# Set device
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Current CUDA Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
decoder = Decoder(OUTPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, device).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

# Optimizer & Loss Function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Load Dataset
dataloader = get_dataloader(DATA_PATH, batch_size=BATCH_SIZE)

# Checkpoint Path
CHECKPOINT_PATH = "checkpoints/"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Resume logic
start_epoch = 0
checkpoint_files = [f for f in os.listdir(CHECKPOINT_PATH) if f.startswith("checkpoint_epoch")]
if checkpoint_files:
    latest_ckpt = sorted(checkpoint_files, key=lambda x: int(x.split("epoch")[1].split(".")[0]))[-1]
    ckpt_path = os.path.join(CHECKPOINT_PATH, latest_ckpt)
    print(f"Resuming from checkpoint: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
else:
    print("No checkpoint found. Starting from scratch.")

# Training Loop
for epoch in range(start_epoch, EPOCHS):
    start_time = time.time()
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(dataloader):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg)

        output = output[1:].reshape(-1, OUTPUT_DIM)
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

        if i % 10 == 0:
            print(f"[Epoch {epoch+1} | Batch {i+1}/{len(dataloader)}] Batch Loss: {loss.item():.4f}")

    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss / len(dataloader):.4f} | Time: {elapsed_time:.2f}s")

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(CHECKPOINT_PATH, f"checkpoint_epoch{epoch+1}.pth"))
    print(f"Checkpoint saved at epoch {epoch+1}")

# Save final model
torch.save(model.state_dict(), "saved_baseline_model.pth")
print("Training complete! Final model saved as 'saved_baseline_model.pth'.")
