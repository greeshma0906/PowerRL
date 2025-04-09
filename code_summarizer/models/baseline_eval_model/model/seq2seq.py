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
        batch_size = trg.shape[0]            # [batch_size, trg_len]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        # Store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encode the source
        encoder_outputs, (hidden, cell) = self.encoder(src)

        # First input to decoder is the <sos> token (first token of each sample)
        input_token = trg[:, 0]  # shape: [batch_size]

        for t in range(1, trg_len):
            # input_token: [batch_size]
            # Pass single timestep input to decoder
            output, hidden, cell, _ = self.decoder(input_token.unsqueeze(1), hidden, cell, encoder_outputs)

            outputs[:, t] = output  # Store prediction at time t

            # Decide next input — teacher forcing or model prediction
            top1 = output.argmax(1)  # [batch_size]
            input_token = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs

