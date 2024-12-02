# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:10:08 2024

@author: micha
"""

import re
import csv
from io import StringIO

def parse_input(text):
    # Define regex patterns for Thought, Action, and Input
    thought_pattern = r"Thought:\s*(.*?)\n"
    action_pattern = r"Action:\s*(.*?)\n"
    input_pattern_list = r"Input:\s*\[(.*?)\]"
    input_pattern_code_1 = r"\s*```python\n(.*?)```"
    input_pattern_code_2 = r"\s*```(.*?)```"
    
    # Extract matches
    thought_match = re.search(thought_pattern, text)
    action_match = re.search(action_pattern, text)
    
    code_list = re.findall(input_pattern_code_1, text, re.DOTALL)
    Extra = ''
    if code_list:
        Extra = 'python\n'
    else:
        code_list = re.findall(input_pattern_code_2, text, re.DOTALL)
    
    print(code_list)
    
    # Extract and clean Thought and Action
    thought = thought_match.group(1).strip() if thought_match else None
    action = action_match.group(1).strip() if action_match else None

    # Temporarily replace code blocks with placeholders
    input_text = text
    for code in code_list:
        input_text = input_text.replace(f"```{Extra}{code}```", "*code")
        
    print(input_text)

    # Extract the Input section
    input_match = re.search(input_pattern_list, input_text, re.DOTALL)
    if not input_match:
        return thought, action, None
    
    # Use csv to correctly parse the list
    input_str = input_match.group(1).strip()
    input_io = StringIO(input_str.replace("\n", ""))  # Ensure newlines are preserved correctly
    reader = csv.reader(input_io, skipinitialspace=True)
    parsed_input = next(reader)  # Parse the single line of CSV into a list
    
    #STRIP EVERYTHING
    for i, item in enumerate(parsed_input):
        parsed_input[i] = item.strip("'\"")
    
    # Replace placeholders with actual code blocks
    for i, item in enumerate(parsed_input):
        if item == "*code":
            parsed_input[i] = code_list.pop(0).strip("'\"")

    return thought, action, parsed_input


def response_formatted(thought, action, input_):
    return '\nAction: ' + action + f'\nInput: {input_}'

# Example usage
response_text = """Action: Write to File
Input: ['main.py', '```python\nfrom knn_classifier.data_loader import load_data\nfrom knn_classifier.knn import KNN\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score\nimport pandas as pd\n\n# Load the data\ntrain_data, train_labels = load_data("./Train_Test_Data/train.csv")\ntest_data, test_labels = load_data("./Train_Test_Data/test.csv")\n\n# Split data into training and validation sets (optional, for better evaluation)\nX_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)\n\n# Initialize and train the KNN classifier\nknn = KNN(k=3)  # You can adjust the value of k\nknn.fit(X_train, y_train)\n\n# Make predictions on the validation set\ny_pred_val = knn.predict(X_val)\nval_accuracy = accuracy_score(y_val, y_pred_val)\nprint(f"Validation Accuracy: {val_accuracy}")\n\n# Make predictions on the test set\ny_pred_test = knn.predict(test_data)\n\n# Create a submission file (optional)\nsubmission = pd.DataFrame({\'label\': y_pred_test})\nsubmission.to_csv(\'./Train_Test_Data/submission_knn.csv\', index=False)\nprint("Submission file created.")\n```', './knn_classifier']
"""
thought, action, input_ = parse_input(response_text)


print(response_formatted(thought, action, input_))

if 'Action:' not in response_text or 'Input:' not in response_text:
    valid = False
else:
    valid = True

import os

class WriteFile():
    def __init__(self, home_dir):
        
        self.home_dir = home_dir
        self.description = """
    Writes the provided content to a file, creating necessary directories if they don't exist.

    Args:
        file_name (str): The name of the file (the file extension will indicate the type of file created e.g. .py, .txt).
        content (str): The content to write into the file.
        directory (str, optional): The relative directory within the home directory where the file should be saved. Defaults to './'.

    Returns:
        str: A status message indicating the file was successfully saved.
        """

    def reWrite_EscapeSequence(self, input_string):
        return input_string.encode('utf-8').decode('unicode_escape')
        
    def forward(self, file_name, content, directory=''):
        
        file_name = file_name.strip()
        
        if directory == '/':
            return "WARN: Use './' not '/' when saving to home directory, file not saved, please try again"
        
        # Construct full directory path
        full_path = os.path.join(self.home_dir, directory)
        os.makedirs(full_path, exist_ok=True)  # Ensure the directory exists
        
        # Full file path
        file_path = os.path.join(full_path, file_name)
        
        new_content = self.reWrite_EscapeSequence(content)
        print(new_content)
        # Write to the file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(new_content)
        
        return "The file was successfully saved."
    
X = WriteFile("D:/Digit_recognizer_AI_stuff")

X.forward(*input_)

