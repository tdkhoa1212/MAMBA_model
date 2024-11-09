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
import os 

# Ignore warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Training and testing the GRAPH_MAMBA model.")
    
    # Arguments for model configurations and training
    parser.add_argument('--weight_path', type=str, default="Weights", help="Path to the model weights")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for the optimizer")
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

# Model configuration
configs = SimpleNamespace(
    seq_len=5,        # Sequence length, L=5
    pred_len=1,        # Prediction length
    num_layers=3,      # R=3
    d_model=64,       # E=64
    d_state=64,        # H=64
    ker_size=2,       
    hidden_dimention=32,  # U=32 
    parallel=False,   
    linear_depth=82, 
    node_num=82,      # N=82
    embed_dim=10,    # de=10
    feature_dim=5,   # L=5
    cheb_k=3         # K=3
)

batch_size = batch_size
processed_data = Get_data(data_path)

for dataset_name, dataset in processed_data.items():
    print('\n' + '-'*30 + f'{dataset_name}' + '-'*30)

    # Initialize the model
    model = GRAPH_MAMBA(configs)
    model.to(device)  # Move model to the GPU if available

    if os.path.exists(f'{weight_path}/{dataset_name}.pth'):
        model.load_state_dict(torch.load(f'{weight_path}/{dataset_name}.pth'))  
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Use learning rate from the arguments
    criterion = nn.MSELoss()
    
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

    num_epochs = epochs
    # lr_patience = 2  
    # best_loss = float('inf')  
    # no_improvement = 0  

    # ----------------------------------------------- Training process -----------------------------------------------
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        with tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100, unit="batch") as train_bar:
            for batch_x, batch_y in train_bar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # Move batch to GPU
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                train_bar.set_postfix(loss=loss.item())
        
        # Validation phase
        model.eval()  
        with torch.no_grad():
            val_loss = 0
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # Move batch to GPU
                val_output = torch.squeeze(model(batch_x))
                loss = criterion(val_output, batch_y)
                val_loss += loss.item()
            
            val_loss /= len(val_loader)

        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     no_improvement = 0
        # else:
        #     no_improvement += 1

        # if no_improvement >= lr_patience:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] /= 2  
        #     no_improvement = 0  

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), f'{weight_path}/{dataset_name}.pth')

    # ----------------------------------------------- Testing process -----------------------------------------------
    model = GRAPH_MAMBA(configs)
    model.load_state_dict(torch.load(f'{weight_path}/{dataset_name}.pth'))
    model.to(device)  # Move model to GPU
    model.eval() 

    true_labels = []
    predictions = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # Move batch to GPU
            batch_output = model(batch_x)  
            true_labels.append(batch_y.cpu().numpy())  # Move to CPU for storage
            predictions.append(batch_output.cpu().numpy())  # Move to CPU for storage
    
    true_labels = np.concatenate(true_labels, axis=0)
    predictions = np.squeeze(np.concatenate(predictions, axis=0))

    # Save testing plot to a specified path
    plt.figure(figsize=(12, 6))
    plt.plot(true_labels, label='True Labels', color='blue', alpha=0.7)
    plt.plot(predictions, label='Predictions', color='red', alpha=0.7)

    plt.title('Model Predictions vs True Labels (Test Data)')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()

    plt.savefig(f'{plot_save_path}/{dataset_name}.png')  # Save plot to the specified path
    plt.close()

    test_loss = RMSE(true_labels, predictions)
    ic_test = information_coefficient(true_labels, predictions)
    ric_test = rank_information_coefficient(true_labels, predictions)
    print(f"Test Loss (RMSE): {test_loss:.4f}")
    print(f"IC: {ic_test:.4f}")
    print(f"RIC: {ric_test:.4f}")
