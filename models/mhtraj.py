import torch
import torch.nn as nn

import numpy as np

from .prt import RT, RTNoEdgeInit
from .my_block import *


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(1024, 512), activation='relu'):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_dims)
        dims.append(output_dim)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class PositionalAgentEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_t_len=200, concat=True):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        if concat:
            self.fc = nn.Linear(2 * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        pe = pe[None].repeat(num_a, 1, 1)
        return pe

    def get_agent_enc(self, num_t, num_a, a_offset):
        ae = self.ae[a_offset: num_a + a_offset, :]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, t_offset=0):
        num_t = x.shape[1]
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset) #(N, T, D)
        if self.concat:
            feat = [x, pos_enc]
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc
        return self.dropout(x) #(N, T, D)


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.decoder_mlp = MLP(
            args.model_dim,
            args.future_length*2,
            hidden_dims=(
                args.model_dim,
                args.model_dim
            )
        )
        # self.decoder_cnn = nn.Conv1d(args.model_dim, args.future_length*2, 3, 1, 1)

    def forward(self, final_feature, cur_location):
        outputs = self.decoder_mlp(final_feature)
        outputs = outputs.view(-1, self.args.future_length, 2)
        if not self.args.pred_rel:
            outputs = outputs + cur_location
        
        return outputs       


class MHTraj(nn.Module):
    def __init__(self, args):
        super(MHTraj, self).__init__()
        self.args = args

        module_args = {
            'num_layers': 1,
            'num_heads': args.num_heads,
            'node_dim': args.model_dim,
            'node_hidden_dim': args.hidden_dim,
            'edge_dim': args.model_dim,
            'edge_hidden_dim_1': args.hidden_dim,
            'edge_hidden_dim_2': args.hidden_dim,
            'dropout': args.dropout,
        }
        
        self.input_dim = len(args.inputs)
        self.input_fc = nn.Linear(self.input_dim, args.model_dim)
        self.input_fc2 = nn.Linear(args.model_dim*args.past_length, args.model_dim)
        self.input_fc3 = nn.Linear(args.model_dim*args.past_length, args.model_dim)
        self.pos_encoder = PositionalAgentEncoding(args.model_dim, 0.1, concat=True)
        self.pair_encoders = nn.ModuleList()
        for i in range(args.num_layers):
            if i == 0:
                self.pair_encoders.append(RT(**module_args))
            else:
                self.pair_encoders.append(RTNoEdgeInit(**module_args))
        
        for i in range(args.sample_k):
            self.add_module("head_%d" % i, Decoder(args))

        self.ffc = FCME(dim=args.model_dim)
        # self.fusion = MultiscaleFusion1D(dim=args.model_dim)

        self.num_temporal_layers = 1
        self.temporal_encoder = nn.ModuleList()
        for i in range(self.num_temporal_layers):
            self.temporal_encoder.append(TCCA(K=8, d=8, kernel_size=3))
        self.lin_ih =nn.Linear(args.model_dim, args.model_dim)
        self.lin_hh =nn.Linear(args.model_dim, args.model_dim)
        self.lin_self =nn.Linear(args.model_dim, args.model_dim)
        
        
    def forward(self, x_abs, x_rel):
        inputs = []
        batch_size, num_agents, length, _ = x_abs.shape
        cur_pos = x_abs[:, :, [-1]].view(batch_size*num_agents, 1, -1).contiguous()

        if 'pos_x' in self.args.inputs and 'pos_y' in self.args.inputs:
            inputs.append(x_abs)
        if 'vel_x' in self.args.inputs and 'vel_y' in self.args.inputs:
            inputs.append(x_rel)
        
        inputs = torch.cat(inputs, dim=-1)
        inputs = inputs.view(batch_size*num_agents, length, -1).contiguous()
        inputs_fc = self.input_fc(inputs).view(batch_size*num_agents, length, self.args.model_dim)  # [N, H, D]
        fft_inputs = self.ffc(inputs_fc.transpose(1, 2)).transpose(1, 2)

        # self.ffc.enable_analysis(True)
        # self.ffc.reset_spectral_stats()

        # self.ffc.visualize_spectrum("spectral_analysis.png")

        inputs_fc = inputs_fc + fft_inputs
        inputs_pos = self.pos_encoder(inputs_fc, num_a=batch_size*num_agents)
        inputs_pos = inputs_pos.view(batch_size, num_agents, length, self.args.model_dim)  # [B, N, H, D]

        inputs_pos_t = inputs_pos.transpose(1, 2)  # [B, H, N, D]
        for i in range(self.num_temporal_layers):
            inputs_pos_t = self.temporal_encoder[i](inputs_pos_t)  # [B, H, N, D]
        inputs_pos_t = inputs_pos_t.transpose(1, 2)  # [B, N, H, D]
        inputs_pos_t = self.input_fc3(inputs_pos_t.contiguous().view(batch_size, num_agents, length*self.args.model_dim))

        n_initial = self.input_fc2(inputs_pos.contiguous().view(batch_size, num_agents, length*self.args.model_dim))
        n_pair, e_pair = n_initial, None
        for i in range(self.args.num_layers):
            n_pair, e_pair = self.pair_encoders[i](n_pair, e_pair, return_edge=True)

        gate = torch.sigmoid(self.lin_ih(n_initial) + self.lin_hh(n_pair))
        n_final = inputs_pos_t + gate * (self.lin_self(n_pair) - inputs_pos_t)
        # n_final = self.fusion(n_pair.transpose(1, 2), n_initial.transpose(1, 2)).transpose(1, 2)

        out_list = []
        for i in range(self.args.sample_k):
            out = self._modules["head_%d" % i](n_final, cur_pos)
            out_list.append(out)
        out = torch.stack(out_list, dim=1)
        out = out.view(batch_size, num_agents, self.args.sample_k, self.args.future_length, -1)
        return out, _
    
