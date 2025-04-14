import pandas as pd
import torch
import torch.nn as nn
import logging
import time
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Set Agg backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np

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

def save_outputs(df, predictions, true_labels, outputs, output_dir='/app/output'):
    """Save predictions, metrics, and ROC curve plot."""
    os.makedirs(output_dir, exist_ok=True)
    df['predicted'] = predictions.numpy()
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
    logger.info(f"Metrics saved to {metrics_path}")
    plt.figure()
    true_labels_one_hot = np.eye(3)[true_labels]
    for i in range(3):
        fpr, tpr, _ = roc_curve(true_labels_one_hot[:, i], outputs.softmax(dim=1).numpy()[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (One-vs-Rest)')
    plt.legend(loc='best')
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    logger.info(f"ROC curve saved to {roc_path}")

def run_inference():
    """Load model and data, run batch inference, and process outputs."""
    try:
        logger.info("Loading inference data from data/inference.csv")
        df = pd.read_csv('/app/data/inference.csv')
        logger.info(f"Inference dataset size: {len(df)} samples")
        X = df.drop('target', axis=1).values
        y_true = df['target'].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        model_path = '/app/models/model.pth'
        logger.info(f"Loading trained model from {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        model = IrisNet()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logger.info("Starting inference")
        start_time = time.time()
        with torch.no_grad():
            outputs = model(X_tensor)
            _, predictions = torch.max(outputs, 1)
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        save_outputs(df, predictions, y_true, outputs)
    except FileNotFoundError as e:
        logger.error(f"File error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    run_inference()