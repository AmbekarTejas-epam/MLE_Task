import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

def prepare_iris_data():
    """
    Loads the Iris dataset, splits it into training and inference sets,
    and saves them as CSVs in the data/ directory.
    """
    try:
        # Load Iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names

        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y

        # Split: 80% train, 20% inference
        train_df, inference_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])

        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)

        # Save to CSVs
        train_df.to_csv('data/train.csv', index=False)
        inference_df.to_csv('data/inference.csv', index=False)

        print(f"Saved {len(train_df)} samples to data/train.csv")
        print(f"Saved {len(inference_df)} samples to data/inference.csv")

    except Exception as e:
        raise Exception(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    prepare_iris_data()