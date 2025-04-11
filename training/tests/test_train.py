import unittest
import pandas as pd
import torch
import os
import shutil
from ..train import IrisNet, train_iris_model  # Relative import

class TestTraining(unittest.TestCase):
    def setUp(self):
        """Set up test data and temporary paths."""
        self.data_path = 'data/train.csv'
        self.model_path = 'models/test_model.pt'
        # Create temporary /app/data/ and /app/models/ structure to match train.py
        os.makedirs('/app/data', exist_ok=True)
        os.makedirs('/app/models', exist_ok=True)  # Added for model saving
        if os.path.exists(self.data_path):
            shutil.copy(self.data_path, '/app/data/train.csv')
        os.makedirs('models', exist_ok=True)  # For local test model path

    def test_data_loading(self):
        """Test if train.csv loads correctly."""
        self.assertTrue(os.path.exists(self.data_path), "train.csv not found")
        df = pd.read_csv(self.data_path)
        self.assertEqual(df.shape[1], 5, "train.csv should have 5 columns")
        self.assertGreater(len(df), 0, "train.csv should not be empty")

    def test_model_training(self):
        """Test if model trains and saves."""
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        train_iris_model()  # Will use /app/data/train.csv and save to /app/models/model.pth
        self.assertTrue(os.path.exists('/app/models/model.pth'), "Model not saved after training")

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists('/app/data/train.csv'):
            os.remove('/app/data/train.csv')
        if os.path.exists('/app/models/model.pth'):
            os.remove('/app/models/model.pth')
        # Remove directories if empty
        try:
            os.rmdir('/app/data')
            os.rmdir('/app/models')
            os.rmdir('/app')
        except OSError:
            pass  # Ignore if directories aren't empty

if __name__ == '__main__':
    unittest.main()