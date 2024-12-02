import os
os.chdir(r'D:/Digit_recognizer_AI_stuff/LLM_Python_Folder/')


from knn_classifier.data_loader import load_data
from knn_classifier.knn import KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the data
train_data, train_labels = load_data("./Train_Test_Data/train.csv")
test_data, test_labels = load_data("./Train_Test_Data/test.csv")

# Split data into training and validation sets (optional, for better evaluation)
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Initialize and train the KNN classifier
knn = KNN(k=3)  # You can adjust the value of k
knn.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = knn.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy: {val_accuracy}")

# Make predictions on the test set
y_pred_test = knn.predict(test_data)

# Create a submission file (optional)
submission = pd.DataFrame({'label': y_pred_test})
submission.to_csv('./Train_Test_Data/submission_knn.csv', index=False)
print("Submission file created.")
