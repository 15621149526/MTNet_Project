import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.mtnet import MTNet
from dataset import RMLDataset
from augmentation import GeneticAugmentation

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_EPOCHS = 100    
    BATCH_SIZE = 128    
    INIT_LR = 5e-4      
    DATA_PATH = 'data/RML2016.10a_dict.pkl'
    
    print(f" MTNet Experiment Started | GPU: {torch.cuda.get_device_name(0)}")

    print(" Loading dataset and splitting train/val sets...")
    train_set = RMLDataset(DATA_PATH, mode='train') 
    val_set = RMLDataset(DATA_PATH, mode='val')     
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = MTNet(num_classes=11).to(device)
    augmenter = GeneticAugmentation(max_epochs=MAX_EPOCHS).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    best_val_acc = 0.0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{MAX_EPOCHS}] Training")
        
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            aug_x = augmenter(batch_x, epoch)
            
            outputs = model(aug_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += pred.eq(batch_y).sum().item()
            
            pbar.set_postfix({'Loss': f"{loss.item():.3f}", 'Acc': f"{100.*train_correct/train_total:.2f}%"})

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for v_x, v_y in val_loader:
                v_x, v_y = v_x.to(device), v_y.to(device)
                v_out = model(v_x) 
                v_loss = criterion(v_out, v_y)
                
                val_loss += v_loss.item()
                _, v_pred = v_out.max(1)
                val_total += v_y.size(0)
                val_correct += v_pred.eq(v_y).sum().item()

        cur_val_acc = 100. * val_correct / val_total
        cur_val_loss = val_loss / len(val_loader)
        
        scheduler.step(cur_val_loss)
        
        print(f" Epoch {epoch} Summary: [Train Acc: {100.*train_correct/train_total:.2f}%] "
              f"[Val Acc: {cur_val_acc:.2f}%] [LR: {optimizer.param_groups[0]['lr']:.6f}]")

        if cur_val_acc > best_val_acc:
            best_val_acc = cur_val_acc
            if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
            torch.save(model.state_dict(), 'checkpoints/mtnet_best.pth')
            print(f" Better model found, saved to checkpoints/mtnet_best.pth")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n🛑 Experiment manually stopped.")
