import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # Read the CSV file into a DataFrame
    data = pd.read_csv("FuelConsumption.csv")

    # the number of times gradient descent will be run and the parameters will be updated
    num_iterations = 500

    # Extract features (X) and labels (Y)
    X = data[['ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION']].to_numpy()
    Y = data['COEMISSIONS '].to_numpy()

    # Split data into training (80%), and test (40%) sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # set up the scaler for the features in the training set
    x_train_scaler = StandardScaler().fit(X_train)
    # set up the scaler for the labels in the training set
    y_train_scaler = StandardScaler().fit(Y_train.reshape(-1, 1))

    # scale the features in the training set
    X_train_scaled = x_train_scaler.transform(X_train)

    # scale the labels in the training set
    Y_train_scaled = y_train_scaler.transform(Y_train.reshape(-1, 1)).flatten()

    # print the dimensions of the data 
    print(f"Shape of X_train: {X_train_scaled.shape}")
    print(f"Shape of Y_train: {Y_train_scaled.shape}")

    # weights of the model
    w = np.zeros(X.shape[1])
    # bias of the model
    b = 0
    # learning rate of the model
    alpha = 0.0001

    # number of training examples
    m = X_train.shape[0]

    # perform the iterations
    for i in range(num_iterations):
        print(f"iteration {i+1}")
        # the total cost as per the current setting of the model
        cost = compute_cost(X_train_scaled, Y_train_scaled, f_wb, w, b)
        # print the parameters and the cost after every 10 iterations
        if (i+1) % 10 == 0:
            print("\n")
            print(f"w: {w}")
            print(f"b: {b:.3f}")
            print(f"cost: {cost:.3f}")
            print("\n")
        # apply gradient descent and get the updated parameters
        w, b = gradient_descent(X_train_scaled, Y_train_scaled, f_wb, w, b, alpha)

    print("\n\n")

    # print the learned parameters
    print(f"w: {w}, b: {b:.3f}")

    # number of test examples
    m_test = X_test.shape[0]

    print("\n\ntest data\n\n")

    # numpy array to contain predictions of the model on test data
    predicted_test = np.zeros(m_test)

    # predict for each test example and write the prediction into the array
    for i in range(m_test):
        predicted_test[i] = f_wb(X_test[i], w, b)

    # transform the predictions back to the original scale
    predicted_test = y_train_scaler.inverse_transform(predicted_test.reshape(-1, 1)).flatten()

    # print the prediction as well as the label for each of the test examples
    for i in range(m_test):
        actual = Y_test[i]
        print(f"{i + 1}) actual: {actual:.3f}, predicted: {predicted_test[i]:.3f}")

    # Calculate accuracy for test data
    accuracy_test = calculate_accuracy(predicted_test, Y_test)

    print("\n\ntrain data\n\n")

    # numpy array to store the predictions of the model on training data
    predicted_train = np.zeros(m)

    # predict for each training example and write the prediction into the array
    for i in range(m):
        predicted_train[i] = f_wb(X_train[i], w, b)

    # transform the predictions back to the original scale
    predicted_train = y_train_scaler.inverse_transform(predicted_train.reshape(-1, 1)).flatten()

    # print the prediction as well as the label for each of the training examples
    for i in range(m):
        actual = Y_train[i]
        print(f"{i + 1}) actual: {actual:.3f}, predicted: {predicted_train[i]:.3f}")

    # Calculate accuracy for train data
    accuracy_train = calculate_accuracy(predicted_train, Y_train)

    print("\n\n")

    print("Accuracy on train data: {:.3f}".format(accuracy_train))
    print("Accuracy on test data: {:.3f}".format(accuracy_test))

# function to evaluate the model's prediction
def f_wb(x, w, b):
    f = np.dot(x, w) + b
    return f


def compute_cost(X, y, f_wb, w, b):
    """
    Compute the cost function of the model.

    Parameters:
    X (numpy.ndarray): Input features.
    y (numpy.ndarray): Target labels.
    f_wb (function): Function to compute the model's output given inputs and parameters.
    w (numpy.ndarray): Model weights.
    b (float): Model bias.

    Returns:
    float: Cost of the model.
    """
    # Initialize cost
    cost = 0

    # Number of training examples
    m = X.shape[0]

    # Compute cost
    for i in range(m):
        cost = cost + (f_wb(X[i], w, b) - y[i]) ** 2

    cost = cost / (2 * m)

    return cost


def compute_gradient(X, y, f_wb, w, b):
    """
    Compute the gradients of the cost function with respect to the weights and bias.

    Parameters:
    X (numpy.ndarray): Input features.
    y (numpy.ndarray): Target labels.
    f_wb (function): Function to compute the model's output given inputs and parameters.
    w (numpy.ndarray): Model weights.
    b (float): Model bias.

    Returns:
    numpy.ndarray, float: Gradients of the cost function with respect to weights and bias.
    """
    # Number of features
    num_features = w.shape[0]

    # Initialize gradients
    dj_dw = np.zeros(num_features)
    dj_db = 0

    # Number of training examples
    m = X.shape[0]

    # Compute gradients
    for j in range(num_features):
        dj_dw_j = 0
        for i in range(m):
            dj_dw_j = dj_dw_j + (f_wb(X[i], w, b) - y[i]) ** 2 * X[i][j]

        dj_dw_j = dj_dw_j / m
        dj_dw[j] = dj_dw_j

    # Compute bias gradient
    for i in range(m):
        dj_db = dj_db + (f_wb(X[i], w, b) - y[i]) ** 2

    dj_db = dj_db / m

    return dj_dw, dj_db





def gradient_descent(X, y, f_wb, w, b, alpha):
    """
    Performs one step of gradient descent to update the weights and bias.

    Parameters:
    X (numpy.ndarray): Input features.
    y (numpy.ndarray): Target labels.
    f_wb (function): Function to compute the model's output given inputs and parameters.
    w (numpy.ndarray): Model weights.
    b (float): Model bias.
    alpha (float): Learning rate.

    Returns:
    numpy.ndarray, float: Updated weights and bias.
    """
    # Number of features
    num_features = w.shape[0]

    # Compute gradients
    dj_dw, dj_db = compute_gradient(X, y, f_wb, w, b)

    # Update weights
    for j in range(num_features):
        w[j] = w[j] - (alpha) * dj_dw[j]

    # Update bias
    b = b - (alpha) * dj_db

    return w, b


def calculate_accuracy(predicted, actual):
    # Calculate absolute errors
    errors = np.abs(predicted - actual)
    # Calculate percentage accuracy
    accuracy = 100 - np.mean(errors / actual) * 100
    return accuracy



if __name__ == "__main__":
    main()

