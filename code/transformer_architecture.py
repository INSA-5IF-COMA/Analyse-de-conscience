import torch.nn as nn
import numpy as np
import torch


# Positional Encoding
def get_sinusoid_encoding_table(positions, d_hid, T=1000, cuda=True):
    ''' Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)'''

    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    
    if cuda:
        return torch.FloatTensor(sinusoid_table).to('cuda')
    else:
        return torch.FloatTensor(sinusoid_table)

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, seq_length=1000, num_heads=5, embed_dim=100): 
        super(Transformer, self).__init__()
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #batch size
        self.hidden_size = hidden_size #hidden size
        self.seq_length = seq_length #sequence length
        self.embed_dim = embed_dim #embedding dimension
        
        positions = seq_length + 1 
        # Positional Encoding
        self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(positions, d_hid=100, T=1000), freeze=True).to('cuda') # frozen weight 
        self.mha = nn.ModuleList()
        self.fc_input_size = embed_dim * seq_length
        self.relu = nn.ReLU()
        # Dropout
        self.dropout = nn.Dropout(p=0.8)
        # LayerNorm 
        self.layernorm = nn.LayerNorm((seq_length, embed_dim))
        self.fc_layernorm = nn.LayerNorm(seq_length * embed_dim)
        
        for _ in range(self.num_layers):
            # Multi-head self-attention Layer
            self.mha.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True))
            # Feed-Forward Network
            # self.encoder.append(nn.Linear(self.fc_input_size, self.fc_input_size))
            
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, 8)
        self.fc3 = nn.Linear(8, 2)
    
    def forward(self,x):
        src_pos = torch.arange(0, self.seq_length, dtype=torch.long).expand(x.shape[0], self.seq_length).to('cuda')
        x = x + self.position_enc(src_pos) # (batch, seq, feature) = (16, 1000, 100)
        for i in range(self.num_layers):
            # attn_output, attn_output_weights (weight = 0 if dropout)
            x2 = self.mha[i](x, x, x)[0] # (query, key, value)
            # residual connection
            x = self.layernorm(x + x2)
            
            #x = x.view(self.input_size, self.fc_input_size) 
            #x3 = x
            #x = self.relu(self.encoder[j](x)) # j = 2*i + 1
            #x = self.dropout(x)
            #x = self.fc_layernorm(x + x3)
            #x = x.view(self.input_size, self.seq_length, self.embed_dim)
            
        x = x.view(-1, self.fc_input_size) 
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x