import torch.nn as nn
from mamba.mamba.mamba import Mamba, MambaBlock
import torch.nn.functional as F


class BidirectionalMambaBlock(nn.Module):
    def __init__(self, configs):
        super(BidirectionalMambaBlock, self).__init__()
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.d_state = configs.d_state
        self.seq_len = configs.seq_len
        self.num_layers = configs.num_layers
        self.parallel = configs.parallel

        self.mamba = MambaBlock(
            d_input=configs.seq_len,
            d_model=configs.d_model,
            d_state=configs.d_state,
            ker_size=configs.ker_size,
            parallel=configs.parallel
        )
        
        self.mamba_reversed = MambaBlock(
            d_input=configs.seq_len,
            d_model=configs.d_model,
            d_state=configs.d_state,
            ker_size=configs.ker_size,
            parallel=configs.parallel
        )

        self.projection_u = nn.Linear(configs.seq_len, configs.hidden_dimention, bias=True)
        self.projection_l = nn.Linear(configs.hidden_dimention, configs.seq_len, bias=True)

        self.norm = nn.LayerNorm(configs.seq_len, eps=1e-5, elementwise_affine=True)
        
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        y1, _ = self.mamba(x)  
        # y1 = self.dropout(y1) 

        x_reversed = x.flip(dims=[1])  
        y2, _ = self.mamba_reversed(x_reversed)
        # y2 = self.dropout(y2) 
        
        y3 = self.norm(x + y1 + y2.flip(dims=[1]))

        y_prime = F.relu(self.projection_u(y3))
        # y_prime = self.dropout(y_prime)  
        y_prime = self.projection_l(y_prime)  
        
        out = self.norm(y_prime + y3)

        return out