# MAMBA_model
'MAMBA MEETS FINANCIAL MARKETS: A GRAPH-MAMBA APPROACH FOR STOCK PRICE PREDICTION' Paper


## Datasets

Download the dataset from [here](https://drive.google.com/drive/folders/1OK8g1Ov-uNpt92S2xVsdZ6vFbvhvZGD_?usp=sharing). After downloading, extract the folder and place the extracted `Data` folder into the `MAMBA_model` directory.

## Setup

1. **Navigate to project directory:**
   ```bash
   %cd MAMBA_model
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
- `--lr`: Learning rate for the optimizer (default: `5e-4`).
- `--epochs`: Number of epochs for training (default: `100`).
- `--batch_size`: Batch size for training (default: `128`).
- `--plot_save_path`: Path to save the testing plot (default: `"Results"`).
- `--data_path`: Path to the data directory (default: `"Data"`).

## Training the Model

To train the model, use the following command:

```bash
python train.py --weight_path <path_to_weights> --lr <learning_rate> --epochs <num_epochs> --batch_size <batch_size> --plot_save_path <plot_save_path> --data_path <data_path>
```

#### Example:

```bash
python train.py --weight_path Weights --lr 5e-4 --epochs 100 --batch_size 128 --plot_save_path Results --data_path Data
```

## Results

| **Dataset** | **Model**               | **RMSE** | **IC**   | **RIC**  | **Parameter Count** |
|-------------|-------------------------|----------|----------|----------|---------------------|
| NASDAQ      | SAMBA (Paper)           | 0.0128   | 0.5046   | 0.4767   | 167,178             |
|             | SAMBA (Implementation)  |          |          |          | 31,189              |
| NYSE        | SAMBA (Paper)           | 0.0125   | 0.5044   | 0.4950   | 167,178             |
|             | SAMBA (Implementation)  |          |          |          | 31,189              |
| DJI         | SAMBA (Paper)           | 0.0108   | 0.4483   | 0.4703   | 167,178             |
|             | SAMBA (Implementation)  |          |          |          | 31,189              |














