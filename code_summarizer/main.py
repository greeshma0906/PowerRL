import torch
import torch.optim as optim
from models.actor_critic import ActorNetwork, CriticNetwork, compute_loss

# Hyperparameters
INPUT_DIM = 256
HIDDEN_DIM = 128
OUTPUT_DIM = 50
NUM_LAYERS = 1
BATCH_SIZE = 32
SEQ_LEN = 20
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4

def train():
    
    actor = ActorNetwork(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)
    critic = CriticNetwork(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS)

    optimizer_actor = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Example batch of states
        state = torch.randn((BATCH_SIZE, SEQ_LEN, INPUT_DIM))  # Random input sequence
        decoder_input = torch.randn((BATCH_SIZE, 1, HIDDEN_DIM))  # Random decoder input
        actions = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,))  # Random actions
        rewards = torch.randn((BATCH_SIZE,))  # Random rewards

        # Compute loss
        actor_loss, critic_loss = compute_loss(actor, critic, state, decoder_input, actions, rewards)

        # Backpropagation
        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        
        actor_loss.backward()
        critic_loss.backward()
        
        optimizer_actor.step()
        optimizer_critic.step()


        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")

if __name__ == "__main__":
    train()
