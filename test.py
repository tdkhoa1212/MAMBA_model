import torch
import matplotlib.pyplot as plt
from Utils.Evaluation_metrics import RMSE, information_coefficient, rank_information_coefficient
from Models.DL_models import GRAPH_MAMBA
from torch.utils.data import DataLoader, TensorDataset
from Utils.Loading_data import processed_data
import numpy as np


batch_size = 64

# Load the saved model
model = GRAPH_MAMBA(configs)
model.load_state_dict(torch.load("Weights/model.pth"))  # Load weights for the model at epoch 5
model.eval()  # Set the model to evaluation mode

# Prepare test data
train_data = processed_data['NASDAQ']
x_test, y_test = torch.tensor(train_data['X_test'], dtype=torch.float32).unsqueeze(-1), torch.tensor(train_data['y_test'], dtype=torch.float32)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize lists to store true labels and predictions
true_labels = []
predictions = []

# Predict on test data
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_output = model(batch_x)  
        true_labels.append(batch_y.numpy())
        predictions.append(batch_output.numpy())

# Flatten the lists of predictions and true labels
true_labels = np.concatenate(true_labels, axis=0)
predictions = np.squeeze(np.concatenate(predictions, axis=0))
print(true_labels.shape, predictions.shape)
# Plotting the predictions vs true labels
plt.figure(figsize=(12, 6))

# Plot the true labels and predictions for the test data
plt.plot(true_labels, label='True Labels', color='blue', alpha=0.7)
plt.plot(predictions, label='Predictions', color='red', alpha=0.7)

plt.title('Model Predictions vs True Labels (Test Data)')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.show()

# Calculate and print evaluation metrics
test_loss = RMSE(true_labels, predictions)
ic_test = information_coefficient(true_labels, predictions)
ric_test = rank_information_coefficient(true_labels, predictions)
print(f"Test Loss (RMSE): {test_loss:.4f}")
print(f"IC: {ic_test:.4f}")
print(f"RIC: {ric_test:.4f}")
