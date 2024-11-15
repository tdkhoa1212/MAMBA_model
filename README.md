# MAMBA_model
![Description of PDF](./Images/conv-mamba.png)

The <strong>Improved GRAPH-MAMBA</strong> model above is in <code>Models.DL_models.GRAPH_MAMBA</code>. 

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

## Experimental Results
### Performance and Parameter Comparison: Graph-Mamba Model (Published Paper) vs. Conv-Graph-Mamba Model (Current Implementation)

| **Dataset** | **Model**                                   | **RMSE ↓** | **IC ↑** | **RIC ↑** | **# Parameters** |
|-------------|--------------------------------------------|------------|----------|-----------|-------------------|
| NASDAQ      | SAMBA (Paper, Graph-Mamba)                 | 0.0128     | 0.5046   | 0.4767    | 167,178           |
|             | SAMBA (Implementation, Conv-Graph-Mamba)   | 0.0096     | 0.2355   | 0.1781    | 34,113            |
| DJI         | SAMBA (Paper, Graph-Mamba)                 | 0.0108     | 0.4483   | 0.4703    | 167,178           |
|             | SAMBA (Implementation, Conv-Graph-Mamba)   | 0.0073     | 0.2156   | 0.2257    | 34,113            |
| NYSE        | SAMBA (Paper, Graph-Mamba)                 | 0.0125     | 0.5044   | 0.4950    | 167,178           |
|             | SAMBA (Implementation, Conv-Graph-Mamba)   |  0.0066    | 0.0904   | 0.1214    | 34,113            |

















