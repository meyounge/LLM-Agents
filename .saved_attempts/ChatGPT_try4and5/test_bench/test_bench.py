import sys
sys.path.append('./')
import digit_classifier

def run_tests():
    # Load the training and test data
    training_data = digit_classifier.load_data('./Train_Test_Data/train.csv')
    test_data = digit_classifier.load_data('./Train_Test_Data/test.csv')

    # Train the model on the training data
    model = digit_classifier.train_model(training_data)

    # Use the model to predict labels for the test data
    predictions = digit_classifier.predict_labels(model, test_data)

    # Calculate the accuracy of the predictions
    accuracy = digit_classifier.calculate_accuracy(predictions, test_data)
    print('Accuracy: ', accuracy)

run_tests()
