import torch
import torch.nn as nn
from model.encoder_decoder import Encoder, Decoder

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]           
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, (hidden, cell) = self.encoder(src)

        input_token = trg[:, 0] 

        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input_token.unsqueeze(1), hidden, cell, encoder_outputs)

            outputs[:, t] = output  
            top1 = output.argmax(1)  
            input_token = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs

