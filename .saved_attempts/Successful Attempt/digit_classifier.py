
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set the working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_datasets():
    train_data = pd.read_csv('./Train_Test_Data/train.csv')
    test_data = pd.read_csv('./Train_Test_Data/test.csv')
    return train_data, test_data

def prepare_data(train_data):
    X = train_data.drop('label', axis = 1)
    y = train_data['label']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_valid, y_train, y_valid

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_valid, y_valid):
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    return accuracy

def predict_test_data(model, test_data):
    test_predictions = model.predict(test_data)
    return test_predictions

def save_predictions(predictions):
    submission = pd.DataFrame({"ImageId": range(1, len(predictions) + 1), "Label": predictions})
    submission.to_csv('./Train_Test_Data/submission.csv', index=False)

def main():
    # Load the data
    train_data, test_data = load_datasets()

    # Prepare the data
    X_train, X_valid, y_train, y_valid = prepare_data(train_data)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    accuracy = evaluate_model(model, X_valid, y_valid)
    print('Model Accuracy:', accuracy)

    # Predict on test data
    predictions = predict_test_data(model, test_data)

    # Save predictions
    save_predictions(predictions)

if __name__ == "__main__":
    main()

