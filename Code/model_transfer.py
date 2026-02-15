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


def fine_tune_model(model, weight_path):
    layers_to_finetune = [
        "temporal_module.lstm.weight_ih_l1", "temporal_module.lstm.weight_hh_l1", "temporal_module.lstm.bias_ih_l1",
        "temporal_module.lstm.bias_hh_l1",
        "temporal_module.conv1_1.weight", "temporal_module.conv1_1.bias",
        "temporal_module.conv2_1.weight", "temporal_module.conv2_1.bias",

        "decoder_1.0.weight", "decoder_1.0.bias",
        "decoder_2.0.weight", "decoder_2.0.bias",
        "decoder_3.0.weight", "decoder_3.0.bias"
    ]

    model.load_state_dict(torch.load(weight_path))

    for name, param in model.named_parameters():
        if name not in layers_to_finetune:
            param.requires_grad = False

    return model


def compare_models(model_path_1, model_path_2):
    state_dict_1 = torch.load(model_path_1, map_location='cpu')
    state_dict_2 = torch.load(model_path_2, map_location='cpu')

    params_1 = state_dict_1 if isinstance(state_dict_1, dict) else state_dict_1.state_dict()
    params_2 = state_dict_2 if isinstance(state_dict_2, dict) else state_dict_2.state_dict()

    different_layers = []
    for name, param1 in params_1.items():
        if name not in params_2:
            print(f"layer {name} do not exist in model 2")
            continue

        param2 = params_2[name]
        if not torch.equal(param1, param2):
            different_layers.append(name)

    if len(different_layers) == 0:
        print("Models share the same parameters")
    else:
        print(f"The following layers have different parameters：")
        for layer in different_layers:
            print(layer)


def transfer_main():
    dataset_B = data_process('./Image pairs B', './sequence_data_B.csv')

    train_loader = DataLoader(dataset_B, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset_B, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset_B, batch_size=1, shuffle=False)

    model = Visual_Module()
    model_path_A = './model_A.pth'
    model_path_B = './model_B.pth'
    model_B = fine_tune_model(model, model_path_A)
    epochs = 100
    criterion = nn.MSELoss()
    optimizer_B = optim.Adam(filter(lambda p: p.requires_grad, model_B.parameters()), lr=1e-4)

    train_model(train_loader, val_loader, model_B, epochs, criterion, optimizer_B, model_path_B)

    compare_models(model_path_A, model_path_B)


if __name__ == '__main__':
    transfer_main()
