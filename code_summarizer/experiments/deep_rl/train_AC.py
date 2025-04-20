import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
import numpy as np
from torch.utils.data import DataLoader,random_split
import sys
import os
import h5py

# Add it to the system path
sys.path.append(os.path.abspath('../../../backend'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('../../utils'))
sys.path.append(os.path.abspath('../../data'))
sys.path.append(os.path.abspath('../../models'))

from config import Config
from custom_dataset import CustomDataset
from deep_rl_summarization.actor_critic import ActorNetwork 
from deep_rl_summarization.actor_critic import CriticNetwork 
from carbontracker.tracker import CarbonTracker


def compute_bleu(reference, predicted):
    """Computes BLEU score for predicted vs. reference summary."""
    reference = [[str(token.item()) if isinstance(token, np.ndarray) else str(token) for token in reference]]
    predicted = [str(token.item()) if isinstance(token, np.ndarray) else str(token) for token in predicted]
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference, predicted, smoothing_function=smoothie)


def update_decoder_input(decoder_input, action_embedding):
    """Updates the decoder input tensor with the latest action (token)."""
    decoder_input = torch.cat((decoder_input[:, 1:, :], action_embedding), dim=1)
    return decoder_input


def train_actor_critic(actor, critic, dataloader, config):
    """Train the actor-critic model with BLEU-based rewards."""
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    if torch.cuda.is_available():
            print("Current CUDA Device:", torch.cuda.current_device())
            print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.critic_learning_rate)
    tracker = CarbonTracker(epochs=config.num_epochs)
    for epoch in range(config.num_epochs):
        tracker.epoch_start()
        actor.train()
        critic.train()
        epoch_actor_loss = 0
        epoch_critic_loss = 0

        for batch_idx, (states, ref_summaries) in enumerate(tqdm(dataloader)):
            states = states.to(config.device).float()

            batch_size = states.shape[0]
            generated_summary = [[] for _ in range(batch_size)]
            decoder_input = torch.zeros((batch_size, 1, actor.encoder.hidden_size * 2), device=config.device)

            for t in range(config.max_summary_length):
                action_logits, _ = actor(states, decoder_input)
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1) 

               
                actions_cpu = action.squeeze(1).cpu().numpy()
                for i in range(batch_size):
                    generated_summary[i].append(actions_cpu[i])

              
                action.float()
                action_embedding = actor.get_action_embedding(action)
                action_embedding.squeeze(1)
                decoder_input = update_decoder_input(decoder_input, action_embedding)

            # Compute BLEU rewards
            rewards = torch.tensor([
                compute_bleu(ref_summaries[i].cpu().numpy(), generated_summary[i])
                for i in range(batch_size)
            ], device=config.device).unsqueeze(1)

            values = critic(states).squeeze(1)

           
            log_probs = torch.stack([
                F.log_softmax(action_logits[i], dim=-1).gather(0, action[i]) for i in range(batch_size)
            ]).unsqueeze(1)

            advantage = rewards - values.detach().unsqueeze(1)

            actor_loss = -torch.mean(log_probs * advantage)
            critic_loss = F.mse_loss(values.unsqueeze(1), rewards)

           
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config.clip_grad)
            critic_optimizer.step()

            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), config.clip_grad)
            actor_optimizer.step()

            epoch_actor_loss += actor_loss.item()
            epoch_critic_loss += critic_loss.item()
        tracker.epoch_end()
       
        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Actor Loss: {epoch_actor_loss / len(dataloader):.3f}')
        print(f'\tTrain Critic Loss: {epoch_critic_loss / len(dataloader):.3f}')
        tracker.stop()

        # Save models
        print(os.getcwd())
        torch.save(actor.state_dict(), f"{config.model_path}/actor_model.pth")
        torch.save(critic.state_dict(), f"{config.model_path}/critic_model.pth")

    return actor, critic

if __name__ == "__main__":
    config = Config()
    # Load Dataset
    print(config.model_path)
    import os
    if not os.path.exists(config.model_path):
        print(f"❌ Directory {config.model_path} does not exist!")
        os.makedirs(config.model_path, exist_ok=True)
        print(f"✅ Created directory: {config.model_path}")
    else:
        print(f"✅ Directory {config.model_path} exists.")
    dataset = CustomDataset("../../dataset/processed_data.h5")  # Replace with your actual dataset path
    with h5py.File("../../dataset/processed_data.h5", "r") as f:
        print("Keys in file:", list(f.keys()))
        print("X shape:", f["X"].shape)
        print("Y shape:", f["Y"].shape)
    print(len(dataset))
    # Split into train/test (e.g., 80/20 split)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Loaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        # Load Config
     # Ensure this contains all needed attributes

    # Initialize Actor and Critic models
    actor = ActorNetwork(config.input_dim,512, 256, config.output_dim, config.num_layers).to(config.device)
    #print(actor)
    critic = CriticNetwork(config.input_dim,512, 256, config.num_layers).to(config.device)

    # Train the Models

    trained_actor, trained_critic = train_actor_critic(actor, critic, train_dataloader, config)

    # Save Final Models
    torch.save(trained_actor.state_dict(), f"{config.model_path}/final_actor.pth")
    torch.save(trained_critic.state_dict(), f"{config.model_path}/final_critic.pth")

    print("Final actor and critic models saved successfully!")
