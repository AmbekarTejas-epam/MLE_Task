import pandas as pd
import torch
import torch.nn as nn
import logging
import time
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IrisNet(nn.Module):
    """Neural network for Iris classification (matches training)."""
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

def run_inference():
    """Load model and data, run batch inference, and save predictions."""
    try:
        # Load inference data
        logger.info("Loading inference data from data/inference.csv")
        df = pd.read_csv('/app/data/inference.csv')
        logger.info(f"Inference dataset size: {len(df)} samples")

        X = df.drop('target', axis=1).values
        X_tensor = torch.FloatTensor(X)

        # Load model
        model_path = '/app/models/model.pth'  # Change to 'model.pth' if not fixed
        logger.info(f"Loading trained model from {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        model = IrisNet()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Run inference
        logger.info("Starting inference")
        start_time = time.time()
        with torch.no_grad():
            outputs = model(X_tensor)
            _, predictions = torch.max(outputs, 1)
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds")

        # Save predictions
        df['predicted'] = predictions.numpy()
        os.makedirs('/app/output', exist_ok=True)
        df.to_csv('/app/output/predictions.csv', index=False)
        logger.info("Predictions saved to output/predictions.csv")

    except FileNotFoundError as e:
        logger.error(f"File error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    run_inference()