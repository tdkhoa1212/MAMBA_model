import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from Utils.Loading_data import datasets, processed_data, feature_names
import numpy as np

def plot_stock_analysis():
    #------------------------ Plot Closing Price Trend for All Datasets ----------------------
    plt.figure(figsize=(12, 6))
    for name, data in datasets.items():
        plt.plot(data['Date'], data['Close'], label=f'{name} Closing Price')

    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Closing Price Trend for All Datasets')
    plt.legend()
    plt.grid()
    plt.show()

    #------------------------ Plot Target Distribution for All Datasets ----------------------
    plt.figure(figsize=(12, 6))
    for i, (name, processed) in enumerate(processed_data.items()):
        ax = plt.subplot(2, 3, i + 1)  # Create a subplot
        sns.countplot(x=processed['y_train'], ax=ax, alpha=0.5)
        ax.set_title(f'{name} Target Distribution')
        ax.set_xlabel('Target (1: Price Increase, 0: Price Decrease)')
        ax.set_ylabel('Count')
    plt.tight_layout()  # Adjust layout
    plt.show()

    #------------------------ Plot Feature Importance for All Datasets ----------------------
    plt.figure(figsize=(15, 10))

    for i, (name, processed) in enumerate(processed_data.items()):
        rf = RandomForestClassifier(random_state=42)
        rf.fit(processed['X_train'], processed['y_train'])
        feature_importances = rf.feature_importances_

        # Get the top 10 feature importances
        sorted_indices = feature_importances.argsort()[-10:]
        top_features = np.array(feature_names[name])[sorted_indices]

        # Create subplot
        ax = plt.subplot(2, 3, i + 1)
        ax.barh(top_features, feature_importances[sorted_indices], alpha=0.7)
        ax.set_title(f'{name} Dataset')
        ax.set_xlabel('Importance')

    # Adjust layout and show plot
    plt.suptitle('Feature Importance for All Datasets')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle
    plt.show()
