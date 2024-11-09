import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
import matplotlib.pyplot as plt


def Get_data(data_path):
    # Load datasets individually
    datasets = {
        'NASDAQ': pd.read_csv(f"{data_path}/Processed_NASDAQ.csv", parse_dates=['Date']),
        'NYSE': pd.read_csv(f"{data_path}/Processed_NYSE.csv", parse_dates=['Date']),
        'S&P': pd.read_csv(f"{data_path}/Processed_SP.csv", parse_dates=['Date']),
        'DJI': pd.read_csv(f"{data_path}/Processed_DJI.csv", parse_dates=['Date']),
        'RUSSELL': pd.read_csv(f"{data_path}/Processed_RUSSELL.csv", parse_dates=['Date'])
    }

    #------------------------ Process each dataset individually ----------------------
    processed_data = {}
    feature_names = {}

    for name, data in datasets.items():
        if 'Name' in data.columns:
            data = data.drop(columns='Name')

        data['return_ratio'] = ((data['Close'].shift(-1) - data['Close']) / data['Close'])*100.
        data.dropna(inplace=True)

        features = data.drop(columns=['Date', 'return_ratio'])
        feature_names[name] = data.columns.tolist()

        # Create rolling windows with a window size of 5 (past 4 days + current day)
        window_size = 5
        features_5day = np.array([features[i:i + window_size].values for i in range(len(features) - window_size + 1)])
        features_5day = features_5day.transpose(0, 2, 1)
        return_ratio_5day = data['return_ratio'].iloc[window_size - 1:].values  # Align labels with windowed data

        #------------------------ Split the data ----------------------
        train_size = int(len(features_5day) * 0.8)
        val_size = int(len(features_5day) * 0.05)

        train_features = np.array(features_5day[:train_size])
        val_features = np.array(features_5day[train_size:train_size + val_size])
        test_features = np.array(features_5day[train_size + val_size:])

        train_labels = np.array(return_ratio_5day[:train_size])
        val_labels = np.array(return_ratio_5day[train_size:train_size + val_size])
        test_labels = np.array(return_ratio_5day[train_size + val_size:])

        #------------------------ Normalize features ----------------------
        scaler = Normalizer()
        train_features = scaler.fit_transform(train_features.reshape(-1, train_features.shape[-1])).reshape(train_features.shape)
        val_features = scaler.transform(val_features.reshape(-1, val_features.shape[-1])).reshape(val_features.shape)
        test_features = scaler.transform(test_features.reshape(-1, test_features.shape[-1])).reshape(test_features.shape)

        #------------------------ Store processed data ----------------------
        processed_data[name] = {
            'X_train': train_features,
            'y_train': train_labels,
            'X_val': val_features,
            'y_val': val_labels,
            'X_test': test_features,
            'y_test': test_labels
        }
    return processed_data

#------------------------ Print shapes of processed data ----------------------
def labels_plotting(processed_data):
    for name, data in processed_data.items():
        print(f'{name} Dataset:')
        print(f'  X_train shape: {data["X_train"].shape}, y_train shape: {data["y_train"].shape}')
        print(f'  X_val shape: {data["X_val"].shape}, y_val shape: {data["y_val"].shape}')
        print(f'  X_test shape: {data["X_test"].shape}, y_test shape: {data["y_test"].shape}')
        
        plt.figure(figsize=(15, 5))

        # Plot y_train
        plt.subplot(1, 3, 1)
        plt.hist(data["y_train"], bins=30, alpha=0.7, color='b')
        plt.title(f'{name} - y_train Distribution')
        plt.xlabel('Target Value')
        plt.ylabel('Frequency')

        # Plot y_val
        plt.subplot(1, 3, 2)
        plt.hist(data["y_val"], bins=30, alpha=0.7, color='g')
        plt.title(f'{name} - y_val Distribution')
        plt.xlabel('Target Value')
        plt.ylabel('Frequency')

        # Plot y_test
        plt.subplot(1, 3, 3)
        plt.hist(data["y_test"], bins=30, alpha=0.7, color='r')
        plt.title(f'{name} - y_test Distribution')
        plt.xlabel('Target Value')
        plt.ylabel('Frequency')

        # Show the plot
        plt.tight_layout()
        plt.show()

