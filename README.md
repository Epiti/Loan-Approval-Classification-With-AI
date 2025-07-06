<<<<<<< HEAD
Loan Approval Classification Dataset

Source: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data?resource=download#


DATA_PREPROCESSING FILE

load_data: Reads data from a CSV file.
preprocess_data:
Handles categorical columns by converting them into dummy variables (one-hot encoding).
Encodes string categorical variables into numeric labels.
Applies a logarithmic transformation to income and loan amount columns to stabilize variance.
Drops the target column (loan_status) from the feature set and concatenates it back at the end.
split_data: Splits the data into training and testing sets, with a specified ratio.


TRADITIONAL_ML FILE

The script defines four different functions to train traditional machine learning models: K-Nearest Neighbors (KNN), Logistic Regression, Decision Tree, and Random Forest.
Each function follows a similar structure:
A model is initialized with some hyperparameters.
The model is trained using model.fit(X_train, y_train).
The trained model is returned for later use (either for predictions or further evaluation).
The make_predictions() function takes a trained model and makes predictions on the given input data.


NEURAL_NETWORKS FILE

Model Definition: I'am defining a simple feedforward neural network using Keras' Sequential model, with 3 layers: an input layer, two hidden layers (with ReLU activation), and an output layer (with sigmoid activation for binary classification).
Compilation: The model is compiled with the Adam optimizer and binary cross-entropy loss, making it suitable for binary classification tasks.
Training: The train_neural_network function trains the model using the provided training data and also tracks the loss values for both the training and validation datasets during training.



MAIN FILE

The script loads and preprocesses the data, then splits it into training and test sets.
It trains and evaluates four traditional machine learning models: K-Nearest Neighbors, Logistic Regression, Decision Tree, and Random Forest.
It trains a neural network and evaluates its performance.
Finally, it plots the training and validation loss curves for the neural network to check for overfitting or underfitting.
=======
# Loan-Approval-Classification-With-AI
>>>>>>> 5c578aab98d21ee9f517a284f8a10a343e70e943
