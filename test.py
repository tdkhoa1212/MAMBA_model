import argparse
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader, TensorDataset
from Utils.Loading_data import Get_data
from Models.DL_models import GRAPH_MAMBA
from Utils.Evaluation_metrics import information_coefficient, rank_information_coefficient, RMSE
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 


warnings.filterwarnings('ignore', category=UserWarning)

configs = SimpleNamespace(
    expand=64,        # H=64 
    pred_len=1,       # Prediction length
    num_layers=3,     # R=3
    d_model=5,       # L=5
    d_state=64,       #  E=64
    seq_len = 5,      # L=5

    hidden_dimention=32,  # U=32
    linear_depth=82, 
    node_num=82,      # N=82
    embed_dim=10,     # de=10
    feature_dim=5,    # L=5
    cheb_k=3          # K=3
)

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

processed_data = Get_data(data_path)

for dataset_name, dataset in processed_data.items():
    x_test = torch.tensor(dataset['X_test'], dtype=torch.float32).to(device)
    y_test = torch.tensor(dataset['y_test'], dtype=torch.float32).to(device)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Testing process
    model = GRAPH_MAMBA(configs)
    model.load_state_dict(torch.load(f'{weight_path}/{dataset_name}_best_ic.pth')) # weights_only=True
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
    plt.figure(figsize=(12, 6))
    plt.plot(true_labels, label='True Labels', color='blue', alpha=0.7)
    plt.plot(predictions, label='Predictions', color='red', alpha=0.7)
    plt.title('Model Predictions vs True Labels (Test Data)')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'{plot_save_path}/{dataset_name}.png')
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