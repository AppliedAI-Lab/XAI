import os
import pandas as pd

datasets_dir = "data/datasets"

for dataset_name in os.listdir(datasets_dir):
      print("\n---", dataset_name)
      dataset_path = os.path.join(datasets_dir, dataset_name)
      train_path = os.path.join(dataset_path, f"{dataset_name}_train.csv")
      test_path = os.path.join(dataset_path, f"{dataset_name}_test.csv")
      train_df = pd.read_csv(train_path)
      test_df = pd.read_csv(test_path)
      print("Total size:", len(train_df) + len(test_df))
      print("Train size:", len(train_df))
      print("Test size:", len(test_df))