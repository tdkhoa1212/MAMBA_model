import torch.nn as nn
# from mamba.mamba import Mamba, MambaBlock
from mamba_ssm import Mamba
import torch.nn.functional as F


class BidirectionalMambaBlock(nn.Module):
    def __init__(self, pred_len, d_model, d_state, seq_len, num_layers, expand, hidden_dimention):
        super(BidirectionalMambaBlock, self).__init__()
        self.pred_len = pred_len
        self.d_model = d_model
        self.d_state = d_state
        self.seq_len = seq_len
        self.num_layers = num_layers

        # self.mamba = MambaBlock(
        #     d_input=seq_len,
        #     d_model=d_model,
        #     d_state=d_state,
        # )
        
        # self.mamba_reversed = MambaBlock(
        #     d_input=seq_len,
        #     d_model=d_model,
        #     d_state=d_state,
        # )

        self.mamba = Mamba(
                            d_model=d_model,  # Model dimension d_model
                            d_state=d_state,  # SSM state expansion factor
                            expand=expand,  # Block expansion factor)
                            d_conv=2
                            )
        self.mamba_reversed = Mamba(
                            d_model=d_model,  # Model dimension d_model
                            d_state=d_state,  # SSM state expansion factor
                            expand=expand,  # Block expansion factor)
                            d_conv=2
                            )

        self.projection_u = nn.Linear(seq_len, hidden_dimention, bias=True)
        self.projection_l = nn.Linear(hidden_dimention, seq_len, bias=True)
        self.l2_lambda = 1e-4
        self.norm = nn.LayerNorm(seq_len, eps=1e-5, elementwise_affine=True)
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        y1 = self.mamba(x)  
    
        x_reversed = x.flip(dims=[1])  
        y2 = self.mamba_reversed(x_reversed)
        
        y3 = self.norm(x + y1 + y2.flip(dims=[1]))

        y_prime = F.relu(self.projection_u(y3))
        y_prime = self.dropout(y_prime)
        y_prime = self.projection_l(y_prime)  
        y_prime = self.dropout(y_prime)
        
        out = self.norm(y_prime + y3)

        return out