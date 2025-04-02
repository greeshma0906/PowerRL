import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1):
        super(ActorNetwork, self).__init__()
        
        # Embedding Layer for categorical inputs
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Encoder (BiLSTM for sequence modeling)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # Attention Mechanism
        self.W_attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.U_attn = nn.Linear(hidden_dim, 1)

        # Decoder (LSTM for sequence prediction)
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first=True)

        # Output Layer (Action Distribution)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, decoder_input):
        # Ensure x is LongTensor for embedding lookup
        max_seq_length = 512  
        x = x[:, :max_seq_length] 
        print(x[:10])  # Check the first few token indices
        x = x.long()
        print(f"Input shape to embedding: {x.shape}")
       # x = self.embedding(x)  

        # Encode the input sequence
        encoder_outputs, (h_n, c_n) = self.encoder(x)

        # Compute attention scores
        attn_scores = self.U_attn(torch.tanh(self.W_attn(encoder_outputs)))  
        attn_weights = F.softmax(attn_scores, dim=1)  # Normalize attention

        # Compute weighted sum of encoder outputs (context vector)
        context_vector = torch.sum(attn_weights * encoder_outputs, dim=1, keepdim=True)  

        # Decode with context vector
        decoder_output, _ = self.decoder(context_vector, (h_n, c_n))

        # Compute action logits
        action_logits = self.fc(decoder_output.squeeze(1))  

        return action_logits, attn_weights  # Return raw logits (for log π computation)


class CriticNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(CriticNetwork, self).__init__()
        
        # Embedding Layer for categorical inputs
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Shared BiLSTM Encoder
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # Attention for Value Estimation
        self.W_attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.U_attn = nn.Linear(hidden_dim, 1)

        # Fully connected output layer (Critic head)
        self.fc = nn.Linear(hidden_dim * 2, 1)  
   
    def forward(self, state):
        # Ensure state is LongTensor for embedding lookup
        state = state.long()
        state = self.embedding(state)  

        encoder_outputs, _ = self.encoder(state)

        # Compute attention scores
        attn_scores = self.U_attn(torch.tanh(self.W_attn(encoder_outputs)))  
        attn_weights = F.softmax(attn_scores, dim=1)

        # Compute context vector
        context_vector = torch.sum(attn_weights * encoder_outputs, dim=1)  

        # Predict value function
        value = self.fc(context_vector)  

        return value  # Return V(s)
