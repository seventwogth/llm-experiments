import numpy as np
import colorsys
import torch
from torch.utils.data import Dataset

def hsv_to_rgb(h, s, v):
    return colorsys.hsv_to_rgb(h, s, v)

def generate_harmonious_palette(n_colors=5):
    h = np.random.rand()
    s = np.random.uniform(0.4, 1.0)
    v = np.random.uniform(0.7, 1.0)
    
    palette = []
    
    scheme = np.random.choice(['analogous', 'monochromatic', 'triadic'])
    
    for i in range(n_colors):
        new_h, new_s, new_v = h, s, v
        
        if scheme == 'monochromatic':
            new_s = max(0, min(1, s + np.random.uniform(-0.3, 0.3)))
            new_v = max(0, min(1, v + np.random.uniform(-0.4, 0.1)))
            
        elif scheme == 'analogous':
            new_h = (h + np.random.uniform(-0.05, 0.05) * i) % 1.0
            
        elif scheme == 'triadic':
            if i > 0:
                new_h = (h + (1/3) * (i % 3)) % 1.0
        
        r, g, b = hsv_to_rgb(new_h, new_s, new_v)
        palette.extend([r, g, b])
        
    return np.array(palette, dtype=np.float32)

class PaletteDataset(Dataset):
    def __init__(self, num_samples=5000):
        self.data = [generate_harmonious_palette() for _ in range(num_samples)]
        self.data = np.array(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])

if __name__ == "__main__":
    ds = PaletteDataset(5)
    print(f"Dataset sample shape: {ds[0].shape}")
    print("Sample palette:", ds[0])
