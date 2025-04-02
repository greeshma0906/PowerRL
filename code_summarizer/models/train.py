import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu

def compute_bleu(reference, predicted):
    """Computes BLEU score for predicted vs. reference summary."""
    return sentence_bleu([reference], predicted)  # BLEU score from NLTK

def train_actor_critic(actor, critic, dataloader, config):
    """Train the actor-critic model with BLEU-based rewards."""
    
    # Initialize optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.critic_learning_rate)
    
    best_valid_bleu = 0

    for epoch in range(config.num_epochs):
        actor.train()
        critic.train()
        epoch_actor_loss = 0
        epoch_critic_loss = 0

        for batch_idx, (states, ref_summaries) in enumerate(tqdm(dataloader)):
            states = states.to(config.device)
            
            generated_summary = []
            decoder_input = torch.zeros((states.shape[0], 1, actor.encoder.hidden_size * 2), device=config.device)

            # Generate summary step-by-step (T timesteps)
            for t in range(config.max_summary_length):
                action_logits, _ = actor(states, decoder_input)  # Get token logits
                action_probs = F.softmax(action_logits, dim=-1)  # Convert to probabilities
                
                action = torch.multinomial(action_probs, 1)  # Sample next token
                generated_summary.append(action.squeeze().cpu().numpy())  # Store generated token
                
                decoder_input = update_decoder_input(decoder_input, action)  # Update state

            # Compute BLEU score as reward
            rewards = torch.tensor([
                compute_bleu(ref_summaries[i].cpu().numpy(), generated_summary[i])
                for i in range(states.shape[0])
            ], device=config.device).unsqueeze(1)

            # Critic estimation (baseline for advantage calculation)
            values = critic(states).squeeze(1)

            # Compute log probabilities of selected actions
            log_probs = torch.stack([
                F.log_softmax(action_logits, dim=-1).gather(1, action).squeeze(1)
                for action_logits, action in zip(action_logits, action)
            ])

            # Compute advantage
            advantage = rewards - values.detach()

            # Compute losses
            actor_loss = -torch.mean(log_probs * advantage)  # Policy loss
            critic_loss = F.mse_loss(values, rewards)  # Value loss

            # Update critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config.clip_grad)
            critic_optimizer.step()

            # Update actor
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), config.clip_grad)
            actor_optimizer.step()

            epoch_actor_loss += actor_loss.item()
            epoch_critic_loss += critic_loss.item()

        # Compute epoch averages
        train_actor_loss = epoch_actor_loss / len(dataloader)
        train_critic_loss = epoch_critic_loss / len(dataloader)

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Actor Loss: {train_actor_loss:.3f}')
        print(f'\tTrain Critic Loss: {train_critic_loss:.3f}')

        # Save the models
        torch.save(actor.state_dict(), f"{config.model_path}/actor_model.pth")
        torch.save(critic.state_dict(), f"{config.model_path}/critic_model.pth")

    return actor, critic
