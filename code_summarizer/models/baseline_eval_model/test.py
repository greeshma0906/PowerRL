import os
import sys
import torch
import h5py
import requests
from torch.utils.data import DataLoader, random_split

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../backend')))

from model.encoder_decoder import Encoder, Decoder
from model.seq2seq import Seq2Seq
from data.custom_dataset import CustomDataset
from utils.metrics import compute_bleu
from carbontracker.tracker import CarbonTracker

# Hyperparameters
INPUT_DIM = 8000
OUTPUT_DIM = 8000
EMBED_SIZE = 256
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.5
BATCH_SIZE = 10
PAD_IDX = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vocab(filepath):
    id2token = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                token = parts[0]
                id2token.append(token)
    return id2token

def decode_indices(indices_batch, vocab_path):
    id2token = load_vocab(vocab_path)
    decoded_sentences = []
    for indices in indices_batch:
        tokens = [id2token[idx] if idx < len(id2token) else '<unk>' for idx in indices]
        sentence = ' '.join(tokens).replace('▁', ' ').strip()
        decoded_sentences.append(sentence)
    return decoded_sentences

def log_bleu_to_logging_server(bleu_score):
    url = 'http://127.0.0.1:5001/log_bleu_rl'
    data = {'bleu_score': bleu_score}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"Successfully logged BLEU score: {bleu_score}")
        else:
            print(f"Failed to log BLEU score: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending BLEU score: {e}")

def evaluate_model(model, dataloader, device, vocab_path):
    model.eval()
    bleu_scores = []

    with torch.no_grad():
        for src, trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg, teacher_forcing_ratio=0)  # no teacher forcing
            predictions = output.argmax(dim=2)

            for i in range(predictions.shape[0]):
                pred = decode_indices([predictions[i].tolist()], vocab_path)[0]
                ref = decode_indices([trg[i].tolist()], vocab_path)[0]

                bleu = compute_bleu(ref.split(), pred.split())
                bleu=bleu*10000
                bleu_scores.append(bleu)
                log_bleu_to_logging_server(bleu)

    return {"BLEU": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0}

if __name__ == "__main__":
    vocab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dataset/bpe_model.vocab"))
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dataset/processed_data.h5"))

    dataset = CustomDataset(dataset_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size])

    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    encoder = Encoder(INPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, DEVICE).to(DEVICE)
    decoder = Decoder(OUTPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, DEVICE).to(DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    checkpoint_path = "saved_baseline_model.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found. Please train the model first.")

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    tracker = CarbonTracker(epochs=1)
    tracker.epoch_start()

    results = evaluate_model(model, test_dataloader, DEVICE, vocab_path)

    tracker.epoch_end()
    tracker.stop()

    print("Baseline Model Evaluation:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
