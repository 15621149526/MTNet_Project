import torch
import torch.nn as nn
import numpy as np

class GeneticAugmentation(nn.Module):
    def __init__(self, max_epochs, alpha=0.4, gamma=5.0):
        super().__init__()
        self.max_epochs = max_epochs
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, current_epoch):
        if not self.training: return x
        
        p_t = self.alpha * (1 - np.exp(-current_epoch / self.max_epochs)) / (1 - np.exp(-self.gamma))
        
        if np.random.rand() < p_t:
            choice = np.random.choice(['reverse', 'shift', 'fragment'])
            if choice == 'reverse':
                x = torch.flip(x, dims=[-1]) 
            elif choice == 'shift':
                shift = np.random.randint(1, x.shape[-1])
                x = torch.roll(x, shifts=shift, dims=-1) 
            elif choice == 'fragment':
                mask_len = np.random.randint(10, 30)
                start = np.random.randint(0, x.shape[-1] - mask_len)
                x[:, :, start:start+mask_len] = 0
        
        mask = torch.bernoulli(torch.full(x.shape, 1 - p_t, device=x.device))
        noise = torch.randn_like(x) * 0.05 
        return x * mask + noise * (1.0 - mask)