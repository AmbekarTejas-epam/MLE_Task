import unittest
import pandas as pd
import torch
import os
import shutil
from ..infer import IrisNet, run_inference  # Relative import from inference/

class TestInference(unittest.TestCase):
    def setUp(self):
        self.data_path = 'data/inference.csv'
        self.model_path = 'models/model.pth'  # Match train.py output
        self.output_path = '/app/output/predictions.csv'  # Match infer.py save path
        # Mimic container paths
        os.makedirs('/app/data', exist_ok=True)
        os.makedirs('/app/models', exist_ok=True)
        os.makedirs('/app/output', exist_ok=True)
        if os.path.exists(self.data_path):
            shutil.copy(self.data_path, '/app/data/inference.csv')
        if os.path.exists(self.model_path):
            shutil.copy(self.model_path, '/app/models/model.pth')

    def test_data_loading(self):
        self.assertTrue(os.path.exists(self.data_path), "inference.csv not found")
        df = pd.read_csv(self.data_path)
        self.assertEqual(df.shape[1], 5, "inference.csv should have 5 columns")
        self.assertGreater(len(df), 0, "inference.csv should not be empty")

    def test_inference_run(self):
        if not os.path.exists('/app/models/model.pth'):
            self.skipTest("Model file not found—run training first")
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        run_inference()
        self.assertTrue(os.path.exists(self.output_path), "Predictions not saved")
        df = pd.read_csv(self.output_path)
        self.assertIn('predicted', df.columns, "Predictions column missing")

    def tearDown(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        if os.path.exists('/app/data/inference.csv'):
            os.remove('/app/data/inference.csv')
        if os.path.exists('/app/models/model.pth'):
            os.remove('/app/models/model.pth')
        try:
            os.rmdir('/app/data')
            os.rmdir('/app/models')
            os.rmdir('/app/output')
            os.rmdir('/app')
        except OSError:
            pass

if __name__ == '__main__':
    unittest.main()