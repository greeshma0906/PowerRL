import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(ActorNetwork, self).__init__()
        
        # Encoder (BiLSTM for sequence processing)
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        
        # Attention parameters
        self.W_attn = nn.Linear(hidden_dim * 2, hidden_dim)  # Project encoder outputs
        self.U_attn = nn.Linear(hidden_dim, 1)  # Compute attention scores
        
        # Decoder LSTM
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, decoder_input):
        """
        x: (batch_size, seq_len, input_dim) -> Input sequence (e.g., code embeddings)
        decoder_input: (batch_size, 1, hidden_dim) -> Decoder hidden state at time t
        """
        batch_size, seq_len, _ = x.shape
        
        # Encode input sequence
        encoder_outputs, (h_n, c_n) = self.encoder(x)  # Shape: (batch_size, seq_len, hidden_dim * 2)
        
        # Compute attention scores
        attn_scores = self.U_attn(torch.tanh(self.W_attn(encoder_outputs)))  # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # Normalize attention weights
        
        # Compute context vector (weighted sum of encoder outputs)
        context_vector = torch.sum(attn_weights * encoder_outputs, dim=1, keepdim=True)  # (batch_size, 1, hidden_dim * 2)
        
        # Decoder step
        decoder_output, _ = self.decoder(context_vector, (h_n, c_n))
        
        # Generate action probabilities
        action_logits = self.fc(decoder_output.squeeze(1))  # (batch_size, output_dim)
        action_probs = F.softmax(action_logits, dim=-1)  # Probabilities over actions
        
        return action_probs, attn_weights

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(CriticNetwork, self).__init__()
        
        # LSTM-based state processing
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        
        # Fully connected layer to predict value function
        self.fc = nn.Linear(hidden_dim * 2, 1)  # Output is a single value vπ(st)
    
    def forward(self, state):
        """
        state: (batch_size, seq_len, input_dim) -> Encoded representation of code snippet
        Returns: (batch_size, 1) -> Predicted value function vπ(st)
        """
        lstm_out, _ = self.lstm(state)  # Shape: (batch_size, seq_len, hidden_dim * 2)
        
        # Use the last hidden state as the representation of the sequence
        final_state = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # Predict the value function
        value = self.fc(final_state)  # (batch_size, 1)
        
        return value