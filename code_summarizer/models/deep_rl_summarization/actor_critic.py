import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1):
        super(ActorNetwork, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
       
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # Attention Mechanism
        self.W_attn = nn.Linear(hidden_dim *2, hidden_dim)
        self.U_attn = nn.Linear(hidden_dim, 1)
        # Decoder (LSTM for sequence prediction)
        self.decoder = nn.LSTM(hidden_dim *2, hidden_dim, num_layers, batch_first=True)

        # Output Layer (Action Distribution)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, decoder_input):
       
        max_seq_length = 8000 
        x = x[:, :max_seq_length] 
        x =  x.long()
        x = self.embedding(x)  
        x = x.float()
        encoder_outputs, (h_n, c_n) = self.encoder(x)

        # Compute attention scores
        attn_scores = self.U_attn(torch.tanh(self.W_attn(encoder_outputs)))  
        attn_weights = F.softmax(attn_scores, dim=1)  

       
        context_vector = torch.sum(attn_weights * encoder_outputs, dim=1, keepdim=True) 
         
        if h_n.shape[0] == 2:  # Means bidirectional=True
            h_n = h_n.mean(dim=0, keepdim=True) 
            c_n = c_n.mean(dim=0, keepdim=True) 
       
        decoder_output, _ = self.decoder(context_vector, (h_n, c_n))

        # Compute action logits
        action_logits = self.fc(decoder_output.squeeze(1))  
        print(f"action_logit shape: {action_logits.shape}")

        return action_logits, attn_weights  
    

    def get_action_embedding(self,action):
        action_embedding = self.embedding(action)
        return action_embedding


class CriticNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(CriticNetwork, self).__init__()
        
       
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        self.W_attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.U_attn = nn.Linear(hidden_dim, 1)

        self.fc = nn.Linear(hidden_dim * 2, 1)  
   
    def forward(self, state):
        state = state.long()
        max_seq_length = 8000 
        state = state[:, :max_seq_length]
        state = self.embedding(state)  

        encoder_outputs, _ = self.encoder(state)

        attn_scores = self.U_attn(torch.tanh(self.W_attn(encoder_outputs)))  
        attn_weights = F.softmax(attn_scores, dim=1)

        
        context_vector = torch.sum(attn_weights * encoder_outputs, dim=1)  

        value = self.fc(context_vector)  

        return value 