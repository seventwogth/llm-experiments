import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from data_gen import PaletteDataset
from models import Generator, Discriminator

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        print("Note: 'torch-directml' not found.")
    except Exception as e:
        print(f"DirectML error: {e}")

    print("Warning: No GPU accelerator found. Using CPU.")
    return torch.device("cpu")

DEVICE = get_device()

# ПАРАМЕТРЫ
LR = 0.0002
BATCH_SIZE = 64
EPOCHS = 50 
Z_DIM = 100 

if not os.path.exists('content'):
    os.makedirs('content')

def visualize_palette(palette_tensor, epoch, idx):
    palette = palette_tensor.cpu().detach().numpy().reshape(5, 3)
    
    palette = np.clip(palette, 0, 1)

    plt.figure(figsize=(8, 2))
    plt.imshow([palette], aspect='auto')
    plt.axis('off')
    plt.title(f'Palette Epoch {epoch}')
    
    filename = f'content/epoch_{epoch}_{idx}.png'
    plt.savefig(filename)
    plt.close()

def train():
    print(f"Starting training on: {DEVICE}")
    start_time = time.time()
    
    # Данные
    dataset = PaletteDataset(num_samples=5000)
    # num_workers=0 важно для Windows, иначе может падать мультипроцессинг
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Модели
    generator = Generator(input_dim=Z_DIM).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    # Оптимизаторы и Лосс
    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=LR)
    opt_d = optim.Adam(discriminator.parameters(), lr=LR)
    
    # Цикл обучения
    for epoch in range(EPOCHS):
        epoch_loss_d = 0.0
        epoch_loss_g = 0.0
        
        for i, real_palettes in enumerate(dataloader):
            real_palettes = real_palettes.to(DEVICE)
            batch_size = real_palettes.size(0)
            
            # --- Обучение Discriminator ---
            opt_d.zero_grad()
            
            # 1. Real Data
            labels_real = torch.ones(batch_size, 1).to(DEVICE)
            output_real = discriminator(real_palettes)
            loss_real = criterion(output_real, labels_real)
            
            # 2. Fake Data
            noise = torch.randn(batch_size, Z_DIM).to(DEVICE)
            fake_palettes = generator(noise)
            labels_fake = torch.zeros(batch_size, 1).to(DEVICE)
            # .detach() нужен, чтобы градиенты не текли в генератор при обучении дискриминатора
            output_fake = discriminator(fake_palettes.detach()) 
            loss_fake = criterion(output_fake, labels_fake)
            
            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()
            epoch_loss_d += loss_d.item()
            
            # --- Обучение Generator ---
            opt_g.zero_grad()
            # Генератор хочет, чтобы дискриминатор сказал "1" (правда) на фейковые данные
            output_fake_for_g = discriminator(fake_palettes)
            loss_g = criterion(output_fake_for_g, labels_real) 
            loss_g.backward()
            opt_g.step()
            epoch_loss_g += loss_g.item()
            
        avg_loss_d = epoch_loss_d / len(dataloader)
        avg_loss_g = epoch_loss_g / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss D: {avg_loss_d:.4f} | Loss G: {avg_loss_g:.4f}")
        
        if (epoch+1) % 5 == 0:
            visualize_palette(fake_palettes[0], epoch+1, 0)
    
    total_time = time.time() - start_time
    print(f"Тренировка завершена за {total_time:.2f} секунд.")
    
    # Сохраняем веса
    torch.save(generator.state_dict(), 'generator.pth')
    print("Модель сохранена в generator.pth")

if __name__ == "__main__":
    train()
