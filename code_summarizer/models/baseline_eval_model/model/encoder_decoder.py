import torch
import torch.nn as nn
import torch.nn.functional as F
from carbontracker.tracker import CarbonTracker

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))  # [batch_size, seq_len, embed_size]
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1).unsqueeze(0)
        cell = torch.cat([cell[-2], cell[-1]], dim=1).unsqueeze(0)
        return outputs, (hidden, cell)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        batch_size, src_len, _ = encoder_outputs.shape
        hidden = hidden.repeat(1, 1, src_len).permute(0, 2, 1)
        hidden = hidden.squeeze(0)  # [batch_size, src_len, hidden_size]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention_weights = F.softmax(attention, dim=1)
        attention_weights = attention_weights.unsqueeze(1)
        context = torch.bmm(attention_weights, encoder_outputs)
        return context, attention_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, embed_size]
        context, attention_weights = self.attention(hidden, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        output = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)
        prediction = self.fc_out(output)
        return prediction, hidden, cell, attention_weights
