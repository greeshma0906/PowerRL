import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(ActorNetwork, self).__init__()
        
        # Encoder (BiLSTM for sequence modeling)
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # Attention Mechanism
        self.W_attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.U_attn = nn.Linear(hidden_dim, 1)

        # Decoder (LSTM for sequence prediction)
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first=True)

        # Output Layer (Action Distribution)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, decoder_input):
        """
        x: (batch_size, seq_len, input_dim) -> Input sequence (code embeddings)
        decoder_input: (batch_size, 1, hidden_dim) -> Decoder's previous hidden state
        """
        batch_size, seq_len, _ = x.shape
        
        # Encode the input sequence
        encoder_outputs, (h_n, c_n) = self.encoder(x)

        # Compute attention scores
        attn_scores = self.U_attn(torch.tanh(self.W_attn(encoder_outputs)))  # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # Normalize attention

        # Compute weighted sum of encoder outputs (context vector)
        context_vector = torch.sum(attn_weights * encoder_outputs, dim=1, keepdim=True)  # (batch_size, 1, hidden_dim * 2)

        # Decode with context vector
        decoder_output, _ = self.decoder(context_vector, (h_n, c_n))

        # Compute action logits
        action_logits = self.fc(decoder_output.squeeze(1))  # (batch_size, output_dim)

        return action_logits, attn_weights  # Return raw logits (for log π computation)

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(CriticNetwork, self).__init__()
        
        # Shared BiLSTM Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # Attention for Value Estimation
        self.W_attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.U_attn = nn.Linear(hidden_dim, 1)

        # Fully connected output layer (Critic head)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # Predicts V(s)
    
    def forward(self, state):
        """
        state: (batch_size, seq_len, input_dim) -> Input sequence representation
        Returns: (batch_size, 1) -> Estimated value function V(s)
        """
        encoder_outputs, _ = self.encoder(state)

        # Compute attention scores
        attn_scores = self.U_attn(torch.tanh(self.W_attn(encoder_outputs)))  # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)

        # Compute context vector
        context_vector = torch.sum(attn_weights * encoder_outputs, dim=1)  # (batch_size, hidden_dim * 2)

        # Predict value function
        value = self.fc(context_vector)  # (batch_size, 1)

        return value  # Return V(s)
