import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import logging
import time
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IrisNet(nn.Module):
    """Neural network for Iris classification."""
    def __init__(self, input_size=4, hidden_size=16, num_classes=3):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_iris_model():
    """Load data, train model, evaluate, and save it."""
    try:
        # Load and preprocess data
        logger.info("Loading training data from data/train.csv")
        df = pd.read_csv('/app/data/train.csv')
        logger.info(f"Dataset size: {len(df)} samples")

        X = df.drop('target', axis=1).values
        y = df['target'].values

        # Split into train and test (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        # Model parameters
        model = IrisNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Train
        logger.info("Starting training")
        start_time = time.time()
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
            logger.info(f"Test accuracy: {accuracy*100:.2f}%")

        # Save model
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), '/app/models/model.pth')

        torch.save(model.state_dict(), 'models/model.pth')
        logger.info("Model saved to models/model.pth")

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_iris_model()