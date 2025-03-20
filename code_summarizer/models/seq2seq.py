import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: [batch_size, src_len]
        trg: [batch_size, trg_len]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encode source sequence
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # First input to the decoder is the <sos> token
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            # Decode
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs)
            
            # Store prediction
            outputs[:, t, :] = output
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token or use ground truth
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs