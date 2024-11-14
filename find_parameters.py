import argparse
from types import SimpleNamespace
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from Utils.Loading_data import Get_data
from Models.DL_models import GRAPH_MAMBA
import tqdm
from Utils.Evaluation_metrics import information_coefficient, rank_information_coefficient, RMSE
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
import itertools
import os
from types import SimpleNamespace

# Ignore warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Training and testing the GRAPH_MAMBA model.")
    parser.add_argument('--weight_path', type=str, default="Weights", help="Path to the model weights")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate for the optimizer")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train the model")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--plot_save_path', type=str, default="Results", help="Path to save the testing plot")
    parser.add_argument('--data_path', type=str, default="Data", help="Path to data")
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set model hyperparameters
weight_path = args.weight_path
lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
plot_save_path = args.plot_save_path
data_path = args.data_path

# Search over `expand` and `d_state` using itertools
hidden_dimention_values = [16, 64, 128, 256]

processed_data = Get_data(data_path)

for dataset_name, dataset in processed_data.items():
    print('\n' + '-' * 30 + f'{dataset_name}' + '-' * 30)
    best_ic = -float('inf')
    best_hidden_dimention = None

    # Iterate over all combinations of expand and d_state
    for hidden_dimention in hidden_dimention_values:
        print(f"Training with hidden_dimention={hidden_dimention}")

        # Model configuration
        configs = SimpleNamespace(
            expand=4,        #  
            pred_len=1,       # Prediction length
            num_layers=6,     # R
            d_model=15,       # N=82
            d_state=64,       # H
            seq_len = 15,      # L=5

            hidden_dimention=hidden_dimention,  # U
            linear_depth=15,   # N=82    
            node_num=15,      # N=82
            embed_dim=30,     # de
            feature_dim=5,    # L=5
            cheb_k=3          # K
        )

        # Load data
        x_train = torch.tensor(dataset['X_train'], dtype=torch.float32).to(device)
        y_train = torch.tensor(dataset['y_train'], dtype=torch.float32).unsqueeze(-1).to(device)
        x_test = torch.tensor(dataset['X_test'], dtype=torch.float32).to(device)
        y_test = torch.tensor(dataset['y_test'], dtype=torch.float32).to(device)

        # Prepare data loaders
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model, optimizer, and criterion
        model = GRAPH_MAMBA(configs).to(device)
        # model.load_state_dict(torch.load(f'{weight_path}/{dataset_name}.pth')) 
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Training loop
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            with tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100, unit="batch") as train_bar:
                for batch_x, batch_y in train_bar:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    train_bar.set_postfix(loss=loss.item())

            # Validation phase to calculate loss and IC
            model.eval()
            val_loss = 0
            true_labels_val, predictions_val = [], []

            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    val_output = model(batch_x)
                    loss = criterion(val_output, batch_y)
                    val_loss += loss.item()
                    true_labels_val.append(batch_y.cpu().numpy())
                    predictions_val.append(val_output.cpu().numpy())

            val_loss /= len(test_loader)
            true_labels_val = np.concatenate(true_labels_val, axis=0)
            predictions_val = np.concatenate(predictions_val, axis=0).squeeze(1)
            current_ic = information_coefficient(true_labels_val, predictions_val)

            # Save the best model based on IC
            if current_ic > best_ic:
                best_ic = current_ic
                hidden_dimention = hidden_dimention
                torch.save(model.state_dict(), f'{weight_path}/{dataset_name}_{hidden_dimention}.pth')
                print(f"New best IC score: {current_ic:.4f} with hidden_dimention={hidden_dimention}. Model weights saved.")

        model.eval()
        true_labels = []
        predictions = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_output = model(batch_x)
                true_labels.append(batch_y.cpu().numpy())
                predictions.append(batch_output.cpu().numpy())
        
        true_labels = np.concatenate(true_labels, axis=0)
        predictions = np.squeeze(np.concatenate(predictions, axis=0))

        # Save testing plot
        plt.figure(figsize=(12, 6))
        plt.plot(true_labels, label='True Labels', color='blue', alpha=0.7)
        plt.plot(predictions, label='Predictions', color='red', alpha=0.7)
        plt.title(f'hidden_dimention={hidden_dimention}')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(f'{plot_save_path}/hidden_dimention={hidden_dimention}.png')
        plt.close()

        test_loss = RMSE(true_labels, predictions)
        ic_test = information_coefficient(true_labels, predictions)
        ric_test = rank_information_coefficient(true_labels, predictions)
        print(f"Test Loss (RMSE): {test_loss:.4f}")
        print(f"IC: {ic_test:.4f}")
        print(f"RIC: {ric_test:.4f}")

        # Save testing results
        results = {
            'Dataset': [dataset_name],
            'RMSE': [test_loss],
            'IC': [ic_test],
            'RIC': [ric_test]
        }
        results_df = pd.DataFrame(results)
        excel_save_path = f'{plot_save_path}/{dataset_name}_{hidden_dimention}.xlsx'
        results_df.to_excel(excel_save_path, index=False)


    