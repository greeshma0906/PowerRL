import torch
import torch.nn as nn
from .encoder_decoder import Encoder, Decoder

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, embed_size, hidden_size, num_layers, dropout):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, embed_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(output_dim, embed_size, hidden_size, num_layers, dropout)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(src.device)
        encoder_outputs, hidden = self.encoder(src)
        input_token = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input_token, hidden, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            input_token = trg[t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs
