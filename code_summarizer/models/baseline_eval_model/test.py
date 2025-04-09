import torch
from model.seq2seq import Seq2Seq
from model.encoder_decoder import Encoder, Decoder
from data_loader import get_dataloader

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Parameters
INPUT_DIM = 5000
OUTPUT_DIM = 5000
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.5
MODEL_PATH = "saved_baseline_model.pth"

encoder = Encoder(INPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
decoder = Decoder(OUTPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

source_texts = ["def multiply(a, b): return a * b"]
target_texts = ["multiplies two numbers"]
source_vocab = {"<pad>": 0, "<unk>": 1, "def": 2, "return": 3}
target_vocab = {"<pad>": 0, "<unk>": 1, "multiplies": 2}

dataloader = get_dataloader(source_texts, target_texts, source_vocab, target_vocab, batch_size=1)


with torch.no_grad():
    for src, src_lengths, trg in dataloader:
        src = src.to(device)
        trg = trg.to(device)

        output = model(src, src_lengths, trg, 0) 
        predicted_tokens = output.argmax(dim=2)  
        
        print("Predicted:", predicted_tokens.tolist())  
