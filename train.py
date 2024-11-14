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
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Import scheduler
import os 

# Ignore warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Training and testing the GRAPH_MAMBA model.")
    
    # Arguments for model configurations and training
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

weight_path = args.weight_path
lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
plot_save_path = args.plot_save_path
data_path = args.data_path

configs = SimpleNamespace(
    expand=4,        #  
    pred_len=1,       # Prediction length
    num_layers=6,     # R
    d_model=15,       # N=82
    d_state=64,       # H

    hidden_dimention=128,  # U
    linear_depth=15,   # N=82    
    node_num=15,      # N=82
    embed_dim=10,     # de
    feature_dim=5,    # L=5
    cheb_k=3          # K
)

processed_data = Get_data(data_path)

for dataset_name, dataset in processed_data.items():
    print('\n' + '-'*30 + f'{dataset_name}' + '-'*30)

    # Initialize the model
    model = GRAPH_MAMBA(configs)
    model.to(device)  

    # if os.path.exists(f'{weight_path}/{dataset_name}.pth'):
    #     model.load_state_dict(torch.load(f'{weight_path}/{dataset_name}.pth'))  
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)   # RMSprop
    criterion = nn.MSELoss()

    # Prepare data loaders
    x_train = torch.tensor(dataset['X_train'], dtype=torch.float32).to(device)
    y_train = torch.tensor(dataset['y_train'], dtype=torch.float32).unsqueeze(-1).to(device)
    x_test = torch.tensor(dataset['X_test'], dtype=torch.float32).to(device)
    y_test = torch.tensor(dataset['y_test'], dtype=torch.float32).to(device)
    x_val = torch.tensor(dataset['X_val'], dtype=torch.float32).to(device)
    y_val = torch.tensor(dataset['y_val'], dtype=torch.float32).to(device)

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"x_val shape: {x_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    best_ic = -float('inf')

    # Training loop
    for epoch in range(epochs):
        if epoch < 1500:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-3
        elif epoch < 2500:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-4
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4

        model.train()
        epoch_loss = 0

        with tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100, unit="batch") as train_bar:
            for batch_x, batch_y in train_bar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad() 
                output = model(batch_x).to(device)
                loss = criterion(output, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                train_bar.set_postfix(loss=loss.item())
        
        # Validation phase to calculate loss and IC
        model.eval()
        val_loss = 0
        true_labels_val = []
        predictions_val = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_output = model(batch_x)
                loss = criterion(val_output, batch_y)
                val_loss += loss.item()
                true_labels_val.append(batch_y.cpu().numpy())
                predictions_val.append(val_output.cpu().numpy())
        val_loss /= len(val_loader)
        true_labels_val = np.concatenate(true_labels_val, axis=0)
        predictions_val = np.concatenate(predictions_val, axis=0).squeeze(1)
        current_ic = information_coefficient(true_labels_val, predictions_val)

        # Save model based on best IC
        if current_ic > best_ic:
            best_ic = current_ic
            torch.save(model.state_dict(), f'{weight_path}/{dataset_name}_best.pth')
            print(f"New best IC score: {current_ic:.4f}. Model weights saved.")

        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {epoch_loss:.2e}, Validation Loss: {val_loss:.2e}")

        torch.save(model.state_dict(), f'{weight_path}/{dataset_name}.pth')

    # Testing process
    model = GRAPH_MAMBA(configs)
    model.load_state_dict(torch.load(f'{weight_path}/{dataset_name}.pth'))
    model.to(device)
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
    plt.figure(figsize=(14, 7))
    
    plt.plot(true_labels, label='True Return Ratio', color='navy', linewidth=2, alpha=0.8)
    plt.plot(predictions, label='Predicted Return Ratio', color='crimson', linestyle='--', linewidth=2, alpha=0.8)
    
    plt.title(f'Model Predictions vs True Labels - {dataset_name}', fontsize=16, fontweight='bold', color='darkslategray')
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('Return Ratio', fontsize=14)
    
    plt.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    
    plt.savefig(f'{plot_save_path}/{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
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
    excel_save_path = f'{plot_save_path}/{dataset_name}_results.xlsx'
    results_df.to_excel(excel_save_path, index=False)
    print(f"Testing results saved to {excel_save_path}")
