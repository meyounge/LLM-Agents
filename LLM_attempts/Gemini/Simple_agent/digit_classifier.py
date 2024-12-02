import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the training data
train_data = pd.read_csv("D:\\Digit_recognizer_AI_stuff\\LLM_Python_Folder\\Train_Test_Data\\train.csv")
X_train = train_data.drop("label", axis=1)
y_train = train_data["label"]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
validation_accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {validation_accuracy}")

# Load the test data
test_data = pd.read_csv("D:\\Digit_recognizer_AI_stuff\\LLM_Python_Folder\\Train_Test_Data\\test.csv")

# Predict on the test data
y_test_pred = model.predict(test_data)

# Create submission file
submission = pd.DataFrame({"ImageId": range(1, len(y_test_pred) + 1), "Label": y_test_pred})
submission.to_csv("submission.csv", index=False)

print("Submission file created successfully.")
