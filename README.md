# MAMBA_model
MAMBA MEETS FINANCIAL MARKETS: A GRAPH-MAMBA APPROACH FOR STOCK PRICE PREDICTION

<strong style="color: red">Note:</strong> The <strong>GRAPH_MAMBA</strong> model, as implemented in this project, is available in the <code>Models.DL_models</code> module. 
## Datasets

Download the `Data` folder from [here](https://drive.google.com/drive/folders/1OK8g1Ov-uNpt92S2xVsdZ6vFbvhvZGD_?usp=sharing). Once downloaded, extract the folder, and place the extracted `Data` folder into the `MAMBA_model`. The final path should be `MAMBA_model/Data`.

## Setup

1. **Navigate to project directory:**
   ```bash
   %cd MAMBA_model
   ```

2. **Install dependencies:**
   ```bash
   !pip install -r requirements.txt
   !pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
   !pip install mamba-ssm==1.2.0.post1
   ```
3. **Testing:** 
   Pre-trained weights are available to test the implemented GRAPH-MAMBA model. Run tests to evaluate its performance, with the test results saved in the `Results` folder.
    ```bash
    !python test.py
    ```
4. **Training:**
    ```bash
    !python train.py
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
















