import pandas as pd
import numpy as np

def load_data(filename):
    df = pd.read_csv(filename)
    X = df.drop("label", axis=1).values
    y = df["label"].values
    return X, y
