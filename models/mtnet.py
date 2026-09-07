import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class SoftThresholding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        gap = x.mean(dim=1, keepdim=True)
        gmp = x.max(dim=1, keepdim=True)[0]
        eta = gap + gmp 
        
        tau = self.net(eta) * gap 
        return torch.sign(x) * torch.relu(torch.abs(x) - tau)


class NMDDA_Block(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.wm = nn.Linear(d_model, d_model, bias=False)

        self.w_r = nn.Parameter(torch.ones(1, 1, d_model))

    def forward(self, x, history_states, x_base=None):
        u_l = self.norm1(x)
        lambda_gate = torch.sigmoid(0.5 * u_l * (1 + torch.erf(u_l / 1.414)))
        
        if len(history_states) > 0:
            hist_sum = sum(history_states)
            hist_proj = self.wm(hist_sum)
            f_agg = (1 - lambda_gate) * x + lambda_gate * hist_proj
        else:
            f_agg = x
            
        attn_out, _ = self.mha(f_agg, f_agg, f_agg)
        attn_out = attn_out + attn_out * self.w_r
        
        residual = x
        if x_base is not None:
            residual = residual + x_base
        psi = self.norm2(attn_out + residual)
        
        return self.mlp(psi) + psi


class MTNet(nn.Module):
    def __init__(self, num_classes=11, d_model=64, max_seq_len=128): 
        super().__init__()
        self.input_proj = nn.Conv1d(2, d_model, kernel_size=7, padding='same')
        
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba1 = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) 
        self.denoise1 = SoftThresholding(d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.mamba2 = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) 
        self.denoise2 = SoftThresholding(d_model)

        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        self.layers = nn.ModuleList([NMDDA_Block(d_model=d_model, nhead=8) for _ in range(4)])
        
        self.agg_proj = nn.Linear(d_model * 2, d_model)
        self.agg_norm = nn.LayerNorm(d_model)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)          
        x = x.transpose(1, 2)           
        
        x_m1 = self.mamba1(self.norm1(x))
        x_d1 = self.denoise1(x_m1)
        x = x + x_d1


        x_m2 = self.mamba2(self.norm2(x))
        x_d2 = self.denoise2(x_m2)
        x = x + x_d2


        x = x + self.pos_emb
        x_denoised = x  
        
        history = []
        for layer in self.layers:
            new_x = layer(x, history, x_base=x_denoised)
            history.append(new_x)  
            x = new_x
            
        avg_feat = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=-1)
        psi = self.agg_norm(self.agg_proj(avg_feat))
        
        return self.classifier(psi)
