import torch
#from your_model_folder.baseline import BaselineModel
#from your_dataloader import test_dataloader, tokenizer
import sys
import h5py
import os
import requests
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('../../../backend'))
from utils.metrics import compute_bleu,compute_rouge
from utils.config import Config
from models.deep_rl_summarization.actor_critic import ActorNetwork
from torch.utils.data import DataLoader,random_split
from data.custom_dataset import CustomDataset
from carbontracker.tracker import CarbonTracker

def load_vocab(filepath):
    """
    Loads vocab file with token-score pairs and returns a list of tokens (id2token).
    """
    id2token = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                token = parts[0]  # The token is the first element in the line
                id2token.append(token)  # Add token to list, index is its position in the list
    
    return id2token

def decode_indices(indices_batch, vocab_path):
    """
    Decodes a batch of token indices into human-readable sentences using the vocab file.
    """
    id2token = load_vocab(vocab_path)  
    
    decoded_sentences = []
    
    for indices in indices_batch:
        tokens = [id2token[idx] if idx < len(id2token) else '<unk>' for idx in indices]
        sentence = ' '.join(tokens).replace('▁', ' ').strip()  
        decoded_sentences.append(sentence)
    
    return decoded_sentences

def log_bleu_to_logging_server(bleu_score):
    """
    Sends the BLEU score to the logging Flask server.
    """
    url = 'http://127.0.0.1:5000/log_bleu_rl'  # URL of the new Flask logging server
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
    tracker = CarbonTracker(epochs=Config.num_epochs)
    
    model.eval()
    bleu_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    # Load the vocab once to be used for decoding
    id2token = load_vocab(vocab_path)
    tracker.epoch_start()
    with torch.no_grad():
        for states, ref_summaries in dataloader:
            
            states = states.to(device).float()
            
            # Predict summaries
            predictions = model.predict(states, Config.max_summary_length)  
            
            for i in range(len(predictions)):
          
                ref = decode_indices([ref_summaries[i]], vocab_path)[0]  
                pred = decode_indices([predictions[i]], vocab_path)[0] 
                
                
                # BLEU
                bleu = compute_bleu(ref.split(), pred.split())  
                bleu_scores.append(bleu)
                log_bleu_to_logging_server(bleu)
                

                # ROUGE
                # rouge_scores = compute_rouge(ref, pred)
                # print(rouge_scores)
                # rouge_1_scores.append(rouge_scores["rouge-1"]["f"])
                # rouge_2_scores.append(rouge_scores["rouge-2"]["f"])
                # rouge_l_scores.append(rouge_scores["rouge-l"]["f"])
    tracker.epoch_end()
    # Calculate average scores
    results = {
        "BLEU": sum(bleu_scores) / len(bleu_scores),
    }
    tracker.stop()

    return results

if __name__ == "__main__":
    device = Config.device  # or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomDataset("../../dataset/processed_data.h5")  # Replace with your actual dataset path
    with h5py.File("../../dataset/processed_data.h5", "r") as f:
        print("Keys in file:", list(f.keys()))
        print("X shape:", f["X"].shape)
        print("Y shape:", f["Y"].shape)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    
    test_dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False)

    
    #baseline_model = BaselineModel().to(device)
    deep_rl_model = ActorNetwork(Config.input_dim,512, 256, Config.output_dim, Config.num_layers).to(Config.device)

    
    #baseline_model.load_state_dict(torch.load("path/to/baseline_model.pth", map_location=device))
    deep_rl_model.load_state_dict(torch.load("models/checkpoints/actor_model.pth", map_location=device))

    # Evaluate
    #baseline_results = evaluate_model(baseline_model, test_dataloader, tokenizer, device)
    deep_rl_results = evaluate_model(deep_rl_model, test_dataloader,device,vocab_path = "../../dataset/bpe_model.vocab")

    print("Baseline Model Evaluation:")
    print("\nDeep RL Model Evaluation:")
    for k, v in deep_rl_results.items():
        print(f"{k}: {v:.4f}")
