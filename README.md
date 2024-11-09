# MAMBA_model
 MAMBA MEETS FINANCIAL MARKETS: A GRAPH-MAMBA APPROACH FOR STOCK PRICE PREDICTION

This repository implements the MAMBA model for stock price prediction, combining state-space models (SSM) with a graph-based approach. The goal is to forecast stock prices using a highly efficient model that leverages MAMBA's computational benefits while incorporating graph structures to improve prediction accuracy.

Here’s a more concise and professional version of the instruction:

## Datasets

Download the dataset from [here](https://www.kaggle.com/datasets/ehoseinz/cnnpred-stock-market-prediction), extract the files, and place the `.csv` files in the `MAMBA_model/Data` folder.

## Setup

1. **Navigate to project directory:**
   ```bash
   %cd /content/MAMBA_model
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Training:**
    ```bash
    python train.py
    ```

## Model Configuration

The model training can be configured via command-line arguments:

- `--weight_path`: Path to save or load model weights (default: `"Weights"`).
- `--lr`: Learning rate for the optimizer (default: `1e-4`).
- `--epochs`: Number of epochs for training (default: `100`).
- `--batch_size`: Batch size for training (default: `128`).
- `--plot_save_path`: Path to save the testing plot (default: `"Results"`).
- `--data_path`: Path to the data directory (default: `"Data"`).

## Training the Model

To train the model, use the following command:

```bash
python train.py --weight_path <path_to_weights> --lr <learning_rate> --epochs <num_epochs> --batch_size <batch_size> --plot_save_path <plot_save_path> --data_path <data_path>
```

### Example:

```bash
python train.py --weight_path Weights --lr 1e-4 --epochs 100 --batch_size 128 --plot_save_path Results --data_path Data
```















