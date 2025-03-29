import torch
import torch.nn as nn
import math

# This code is originally written in https://github.com/jamqd/Group-Preference-Optimization

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class TNP(nn.Module):
    def __init__(
        self,
        dim_x,
        dim_y,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers,
        bound_std
    ):
        super(TNP, self).__init__()

        #a simple MLP to embed the input, take input a word and the out is a vector of size d_model (128)
        self.embedder = self.build_mlp(dim_x + dim_y, d_model, d_model, emb_depth)

        # Transformer Encoder Layer is a stack of n layers of self-attention and feedforward network layers
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        # Transformer Encoder is composed of a stack of N=6 identical layers (num_layers) 
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.bound_std = bound_std



    def construct_input(self, batch, autoreg=False):
        
        x_y_ctx = torch.cat((batch['xc'], batch['yc']), dim=-1)
        x_0_tar = torch.cat((batch['xt'], torch.zeros_like(batch['yt'])), dim=-1)
        if not autoreg:
            inp = torch.cat((x_y_ctx, x_0_tar), dim=1)
        else:
            if self.training and self.bound_std:
                yt_noise = batch['yt'] + 0.05 * torch.randn_like(batch['yt']) # add noise to the past to smooth the model
                x_y_tar = torch.cat((batch['xt'], yt_noise), dim=-1)
            else:
                x_y_tar = torch.cat((batch['xt'], batch['yt']), dim=-1)
            inp = torch.cat((x_y_ctx, x_y_tar, x_0_tar), dim=1)
        return inp

    def create_mask(self, batch, autoreg=False):
        num_ctx = batch['xc'].shape[1]
        num_tar = batch['xt'].shape[1]
        num_all = num_ctx + num_tar

       # Create source key padding mask [batch_size, sequence_length]
        padding_mask_ctx = (torch.sum(batch['xc'], dim=-1) == 0)
        padding_mask_tar = (torch.sum(batch['xt'], dim=-1) == 0)

        src_key_padding_mask = torch.cat([padding_mask_ctx, padding_mask_tar], dim=1)
        if not autoreg:
            mask = torch.zeros(num_all, num_all, device='cuda').fill_(float('-inf'))
            mask[:, :num_ctx] = 0.0
        else:
            mask = torch.zeros((num_all+num_tar, num_all+num_tar), device='cuda').fill_(float('-inf'))
            mask[:, :num_ctx] = 0.0 # all points attend to context points
            mask[num_ctx:num_all, num_ctx:num_all].triu_(diagonal=1) # each real target point attends to itself and precedding real target points
            mask[num_all:, num_ctx:num_all].triu_(diagonal=0) # each fake target point attends to preceeding real target points
            src_key_padding_mask = torch.cat([padding_mask_ctx, padding_mask_tar, padding_mask_tar], dim=1)
        return mask, src_key_padding_mask, num_tar

    def encode(self, batch, autoreg=False):
        inp = self.construct_input(batch, autoreg)
        mask, src_key_padding_mask, num_tar = self.create_mask(batch, autoreg)
        embeddings = self.embedder(inp)
        out = self.encoder(embeddings, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return out[:, -num_tar:]
    
    def build_mlp(self, dim_in, dim_hid, dim_out, depth):
        modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
        for _ in range(depth-2):
            modules.append(nn.Linear(dim_hid, dim_hid))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dim_hid, dim_out))
        return nn.Sequential(*modules)
