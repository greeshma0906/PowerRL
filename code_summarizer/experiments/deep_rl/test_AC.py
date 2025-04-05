import torch
from utils.metrics import compute_bleu,compute_rouge
from your_model_folder.baseline import BaselineModel
from your_model_folder.deep_rl import DeepRLModel
from your_dataloader import test_dataloader, tokenizer
import config 

def evaluate_model(model, dataloader, tokenizer, device):
    model.eval()
    bleu_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    with torch.no_grad():
        for states, ref_summaries in dataloader:
            states = states.to(device).float()

            # Predict summaries
            predictions = model.predict(states)  # should return token indices or decoded tokens

            for i in range(len(predictions)):
                # Tokenize reference and prediction if needed
                ref = [str(token) for token in ref_summaries[i]]
                pred = [str(token) for token in predictions[i]]

                # BLEU
                bleu = compute_bleu(ref, pred)
                bleu_scores.append(bleu)

                # ROUGE
                rouge_scores = compute_rouge(ref, pred)
                rouge_1_scores.append(rouge_scores["rouge-1"]["f"])
                rouge_2_scores.append(rouge_scores["rouge-2"]["f"])
                rouge_l_scores.append(rouge_scores["rouge-l"]["f"])

    results = {
        "BLEU": sum(bleu_scores) / len(bleu_scores),
        "ROUGE-1": sum(rouge_1_scores) / len(rouge_1_scores),
        "ROUGE-2": sum(rouge_2_scores) / len(rouge_2_scores),
        "ROUGE-L": sum(rouge_l_scores) / len(rouge_l_scores)
    }

    return results

if __name__ == "__main__":
    device = config.device  # or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomDataset("../../dataset/processed_data.h5")  # Replace with your actual dataset path
    with h5py.File("../../dataset/processed_data.h5", "r") as f:
        print("Keys in file:", list(f.keys()))
        print("X shape:", f["X"].shape)
        print("Y shape:", f["Y"].shape)
    # Split into train/test (e.g., 80/20 split)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Loaders
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize your models
    baseline_model = BaselineModel().to(device)
    deep_rl_model = DeepRLModel().to(device)

    # Load model weights if needed
    baseline_model.load_state_dict(torch.load("path/to/baseline_model.pth", map_location=device))
    deep_rl_model.load_state_dict(torch.load("path/to/deep_rl_model.pth", map_location=device))

    # Evaluate
    baseline_results = evaluate_model(baseline_model, test_dataloader, tokenizer, device)
    deep_rl_results = evaluate_model(deep_rl_model, test_dataloader, tokenizer, device)

    print("Baseline Model Evaluation:")
    for k, v in baseline_results.items():
        print(f"{k}: {v:.4f}")

    print("\nDeep RL Model Evaluation:")
    for k, v in deep_rl_results.items():
        print(f"{k}: {v:.4f}")