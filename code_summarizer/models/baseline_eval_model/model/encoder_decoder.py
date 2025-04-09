import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1, device="cpu"):
        super(Encoder, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        max_seq_len = 100
        x = src[:, :max_seq_len].long().to(self.device)
        #print("Input shape:", x.shape)  # [batch, seq_len]

        embedded = self.dropout(self.embedding(x))
        #print("Embedded shape:", embedded.shape)  # [batch, seq_len, embed_size]

        outputs, (hidden, cell) = self.lstm(embedded)
        #print("LSTM outputs shape:", outputs.shape)  # [batch, seq_len, hidden_size*2]
        #print("LSTM hidden shape:", hidden.shape)    # [2, batch, hidden_size]
        #print("LSTM cell shape:", cell.shape)        # [2, batch, hidden_size]

        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (B, 1024)
        #print("Concatenated hidden_cat shape:", hidden_cat.shape)

        hidden_cat = hidden_cat.unsqueeze(0).repeat(2, 1, 1)     # (2, B, 1024)
        #print("Repeated hidden_cat shape:", hidden_cat.shape)

        cell_cat = torch.cat((cell[-2], cell[-1]), dim=1)        # (B, 1024)
        #print("Concatenated cell_cat shape:", cell_cat.shape)

        cell_cat = cell_cat.unsqueeze(0).repeat(2, 1, 1)         # (2, B, 1024)
        #print("Repeated cell_cat shape:", cell_cat.shape)

        return outputs, (hidden_cat, cell_cat)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 4, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        #print("Attention hidden shape:", hidden.shape)            # [num_layers, batch, hidden*2]
        #print("Encoder outputs shape:", encoder_outputs.shape)    # [batch, seq_len, hidden*2]

        batch_size, src_len, _ = encoder_outputs.shape

        hidden = hidden.permute(1, 0, 2)  # [batch, num_layers, hidden*2]
        hidden = hidden[:, -1, :]         # [batch, hidden*2]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch, src_len, hidden*2]
        #print("Expanded hidden for attention:", hidden.shape)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch, src_len, hidden]
        #print("Energy shape:", energy.shape)

        attention = self.v(energy).squeeze(2)  # [batch, src_len]
        attention_weights = F.softmax(attention, dim=1)  # [batch, src_len]
        #print("Attention weights shape:", attention_weights.shape)

        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden*2]
        #print("Context shape:", context.shape)

        return context, attention_weights.unsqueeze(1)  # [batch, 1, src_len]


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1, device="cpu"):
        super(Decoder, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(
            embed_size + hidden_size*2,
            hidden_size*2,
            num_layers=2,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_size * 4, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        #print("Decoder input shape:", input.shape)  # [batch]
        embedded = self.dropout(self.embedding(input.to(self.device)))  # [batch, 1, embed_size]
        #print("Embedded decoder input shape:", embedded.shape)

        context, attention_weights = self.attention(hidden, encoder_outputs)  # context: [batch, 1, hidden*2]
        #print("Context vector shape:", context.shape)

        rnn_input = torch.cat((embedded, context), dim=2)  # [batch, 1, embed + hidden*2]
        #print("RNN input shape:", rnn_input.shape)

        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        #print("Decoder LSTM output shape:", output.shape)
        #print("Decoder hidden shape:", hidden.shape)
        #print("Decoder cell shape:", cell.shape)

        output = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)  # [batch, hidden*3]
       # print("Concat output+context shape:", output.shape)

        prediction = self.fc_out(output)  # [batch, vocab_size]
       # print("Final prediction shape:", prediction.shape)

        return prediction, hidden, cell, attention_weights

