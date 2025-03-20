import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import os

from data.dataset import CodeCommentDataset, collate_fn
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from models.actor_critic import ActorCritic
from utils.metrics import bleu_score, perplexity
from utils.config import Config

def train_seq2seq(model, train_loader, valid_loader, criterion, optimizer, config):
    """Train the sequence-to-sequence model."""
    best_valid_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (src, trg) in enumerate(tqdm(train_loader)):
            src = src.to(config.device)
            trg = trg.to(config.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(src, trg)
            
            # Reshape output and target for loss calculation
            # output: [batch_size, trg_len, output_dim]
            # trg: [batch_size, trg_len]
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            # Calculate loss
            loss = criterion(output, trg)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            
            # Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_loader)
        
        # Validate
        valid_loss = evaluate(model, valid_loader, criterion, config)
        
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tValid Loss: {valid_loss:.3f}')
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(config.model_path, 'seq2seq_best.pt'))
    
    return model

def train_actor_critic(model, train_loader, valid_loader, criterion, config):
    """Train the actor-critic model."""
    # Initialize optimizers
    actor_optimizer = optim.Adam(model.seq2seq.parameters(), lr=config.actor_learning_rate)
    critic_optimizer = optim.Adam(model.critic.parameters(), lr=config.critic_learning_rate)
    
    best_valid_bleu = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        epoch_bleu = 0
        
        for batch_idx, (src, trg) in enumerate(tqdm(train_loader)):
            src = src.to(config.device)
            trg = trg.to(config.device)
            
            # Forward pass through actor
            outputs, generated_tokens, _ = model.forward_actor(src)
            
            # Convert to words for BLEU calculation
            generated_seqs = []
            reference_seqs = []
            
            for i in range(len(src)):
                gen_seq = [token.item() for token in generated_tokens[i]]
                # Remove padding, <sos>, and <eos>
                gen_seq = [t for t in gen_seq if t > 3]
                
                ref_seq = [token.item() for token in trg[i]]
                # Remove padding, <sos>, and <eos>
                ref_seq = [t for t in ref_seq if t > 3]
                
                generated_seqs.append(gen_seq)
                reference_seqs.append(ref_seq)
            
            # Calculate BLEU score
            batch_bleu = bleu_score(reference_seqs, generated_seqs)
            epoch_bleu += batch_bleu
            
            # Calculate value predictions
            values = model.forward_critic(src, generated_tokens)
            
            # Calculate advantage (reward - value)
            rewards = torch.tensor([batch_bleu] * len(src)).to(config.device).unsqueeze(1)
            advantages = rewards - values
            
            # Calculate critic loss
            critic_loss = nn.MSELoss()(values, rewards)
            
            # Calculate actor loss
            actor_loss = 0
            for t in range(1, generated_tokens.shape[1]):
                if t == 0:
                    continue
                
                log_probs = F.log_softmax(outputs[:, t], dim=1)
                selected_log_probs = log_probs.gather(1, generated_tokens[:, t].unsqueeze(1))
                actor_loss += -(selected_log_probs * advantages.detach()).mean()
            
            # Backward pass for critic
            critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), config.clip_grad)
            critic_optimizer.step()
            
            # Backward pass for actor
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.seq2seq.parameters(), config.clip_grad)
            actor_optimizer.step()
            
            epoch_actor_loss += actor_loss.item()
            epoch_critic_loss += critic_loss.item()
        
        train_actor_loss = epoch_actor_loss / len(train_loader)
        train_critic_loss = epoch_critic_loss / len(train_loader)
        train_bleu = epoch_bleu / len(train_loader)
        
        # Validate
        valid_bleu = evaluate_actor_critic(model, valid_loader, config)
        
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Actor Loss: {train_actor_loss:.3f}')
        print(f'\tTrain Critic Loss: {train_critic_loss:.3f}')
        print(f'\tTrain BLEU: {train_bleu:.3f}')
        print(f'\tValid BLEU: {valid_bleu:.3f}')
        
        # Save best model
        if valid_bleu > best_valid_bleu:
            best_valid_bleu = valid_bleu
            torch.save(model.state_dict(), os.path.join(config.model_path, 'actor_critic_best.pt'))
    
    return model

def evaluate(model, data_loader, criterion, config):
    """Evaluate the sequence-to-sequence model."""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, trg in data_loader:
            src = src.to(config.device)
            trg = trg.to(config.device)
            
            # Forward pass
            output = model(src, trg, 0)  # Turn off teacher forcing
            
            # Reshape output and target for loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            # Calculate loss
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)

def evaluate_actor_critic(model, data_loader, config):
    """Evaluate the actor-critic model."""
    model.eval()
    epoch_bleu = 0
    
    with torch.no_grad():
        for src, trg in data_loader:
            src = src.to(config.device)
            trg = trg.to(config.device)
            
            # Forward pass through actor
            _, generated_tokens, _ = model.forward_actor(src)
            
            # Convert to words for BLEU calculation
            generated_seqs = []
            reference_seqs = []
            
            for i in range(len(src)):
                gen_seq = [token.item() for token in generated_tokens[i]]
                # Remove padding, <sos>, and <eos>
                gen_seq = [t for t in gen_seq if t > 3]
                
                ref_seq = [token.item() for token in trg[i]]
                # Remove padding, <sos>, and <eos>
                ref_seq = [t for t in ref_seq if t > 3]
                
                generated_seqs.append(gen_seq)
                reference_seqs.append(ref_seq)
            
            # Calculate BLEU score
            batch_bleu = bleu_score(reference_seqs, generated_seqs)
            epoch_bleu += batch_bleu
    
    return epoch_bleu / len(data_loader)

