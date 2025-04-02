import torch
from torch.utils.data import DataLoader
from utils.config import Config  # Ensure you have a config class or dictionary
from models.actor_critic import ActorNetwork  # Ensure your actor model class is implemented
from models.actor_critic import CriticNetwork  # Ensure your critic model class is implemented
from data.custom_dataset import CustomDataset  # Your dataset class that loads .h5 data
from models.train import train_actor_critic
# Load Dataset
dataset = CustomDataset("dataset/processed_data.h5")  # Replace with your actual dataset path
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load Config
config = Config()  # Ensure this contains all needed attributes

# Initialize Actor and Critic models
actor = ActorNetwork(config.input_dim, config.hidden_dim, config.output_dim, config.num_layers).to(config.device)
critic = CriticNetwork(config.input_dim, config.hidden_dim, config.num_layers).to(config.device)

# Train the Models
trained_actor, trained_critic = train_actor_critic(actor, critic, dataloader, config)

# Save Final Models
torch.save(trained_actor.state_dict(), f"{config.model_path}/final_actor.pth")
torch.save(trained_critic.state_dict(), f"{config.model_path}/final_critic.pth")

print("Final actor and critic models saved successfully!")