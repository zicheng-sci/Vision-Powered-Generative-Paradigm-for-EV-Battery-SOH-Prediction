import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from model import Visual_Module
from data_prepare import data_process

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(train_loader, val_loader, model, epochs, criterion, optimizer, save_path):
    train_loss = []
    val_loss = []
    model.to(device)
    for epoch in range(epochs):
        model.train()
        batch_loss = []
        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        for img_in, seq_in, targets in train_loader_tqdm:
            img_in, seq_in, targets = img_in.to(device), seq_in.to(device), targets.to(device)
            outputs = model(img_in, seq_in)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        train_loss.append(np.average(batch_loss))

        model.eval()
        val_batch_loss = []
        with torch.no_grad():
            for img_in, seq_in, targets in val_loader:
                img_in, seq_in, targets = img_in.to(device), seq_in.to(device), targets.to(device)
                outputs = model(img_in, seq_in)
                loss = criterion(outputs, targets)
                val_batch_loss.append(loss.item())

        val_loss.append(np.average(val_batch_loss))

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}')

    torch.save(model.state_dict(), save_path)


def pretrain_main():
    dataset_A = data_process('./Image pairs A', './sequence_data_A.csv')

    train_loader = DataLoader(dataset_A, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset_A, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset_A, batch_size=1, shuffle=False)

    model_A = Visual_Module()
    model_path_A = './model_A.pth'
    epochs = 100
    criterion = nn.MSELoss()
    optimizer_A = optim.Adam(filter(lambda p: p.requires_grad, model_A.parameters()), lr=1e-4)

    train_model(train_loader, val_loader, model_A, epochs, criterion, optimizer_A, model_path_A)


if __name__ == '__main__':
    pretrain_main()
