import numpy as np

# import rasterio

import sys

# import os
from sklearn.model_selection import train_test_split
from . import helpers

# from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, fbeta_score, precision_recall_curve
from matplotlib import pyplot

# from mlxtend.preprocessing import minmax_scaling
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc

# Currently ALL AT UP TO 60% NAN
TRAINING_DATASET_FOLDERNAMES = [
    "Bahraich-buffer-5-miles",
    "Barabanki-buffer-5-miles",
    "Kanpur-buffer-5-miles",
    "Sitapur-buffer-5-miles",
    "Varanasi-buffer-5-miles",
]

TEST_SIZE = 0.3
CLASS_MAPPING = {"HIGH": 1, "LOW": 0}


def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title("Cross Entropy Loss")
    pyplot.plot(history.history["loss"], color="blue", label="train")
    pyplot.plot(history.history["val_loss"], color="orange", label="test")
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title("Fbeta")
    pyplot.plot(history.history["fbeta"], color="blue", label="train")
    pyplot.plot(history.history["val_fbeta"], color="orange", label="test")
    # plot accuracy
    # pyplot.subplot(211)
    # pyplot.title("accuracy")
    # pyplot.plot(history.history["accuracy"], color="blue", label="train")
    # pyplot.plot(history.history["val_accuracy"], color="orange", label="test")

    # save plot to file
    filename = sys.argv[0].split("/")[-1]
    pyplot.savefig(filename + "_plot.png")
    pyplot.close()


blue = "#1f77b4"
red = "#d62728"
grey = "grey"
green = "green"

from sklearn.metrics import ConfusionMatrixDisplay


def run_test_harness_logistic_regression(trainX, trainY, testX, testY):
    # define model
    # model = LogisticRegressionCV(scoring=["accuracy", "neg_log_loss", "roc_auc"])
    model = LogisticRegressionCV(scoring="accuracy", solver="liblinear")
    # linear regression
    # polynomial regression

    # fit model
    model.fit(trainX, trainY)

    # predict
    predictions = model.predict(testX)
    accuracy = accuracy_score(testY, predictions)
    print("Accuracy:", accuracy)

    print("Actual labels length", len(testY))
    print("Actual labels LOW count", np.count_nonzero(testY == CLASS_MAPPING["LOW"]))
    print("Actual labels HIGH count", np.count_nonzero(testY == CLASS_MAPPING["HIGH"]))

    print("Prediction labels length", len(predictions))
    print("Prediction labels LOW count", np.count_nonzero(predictions == CLASS_MAPPING["LOW"]))
    print("Prediction labels HIGH count", np.count_nonzero(predictions == CLASS_MAPPING["HIGH"]))

    # ----- Calculate the ROC curve and AUC GOOD FOR THESIS. --------
    y_prob = model.predict_proba(testX)[:, CLASS_MAPPING["HIGH"]]  # Probability of class 1 (YES)

    fpr, tpr, thresholds = roc_curve(testY, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    # pyplot.figure(figsize=(8, 6))
    # pyplot.plot(fpr, tpr, color=red, lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    # pyplot.plot([0, 1], [0, 1], color=grey, lw=2, linestyle="--")
    # pyplot.xlim([0.0, 1.0])
    # pyplot.ylim([0.0, 1.05])
    # pyplot.xlabel("False Positive Rate")
    # pyplot.ylabel("True Positive Rate")
    # pyplot.title("Receiver Operating Characteristic (ROC)")
    # pyplot.legend(loc="lower right")
    # pyplot.show()

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(testY, predictions)
    # Plot the confusion matrix
    # cm_display = ConfusionMatrixDisplay(conf_matrix).plot()
    # cm_display.plot()
    # pyplot.show()

    # For more accurate plotting:
    # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py

    TP = conf_matrix[0][0]
    TN = conf_matrix[1][1]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    specificity = TN / (TN + FP)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1_score)
    print("Specificity:", specificity)

    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate and print ROC AUC score
    print("ROC AUC Score:", roc_auc_score(testY, y_prob))

    # What's the loss
    loss = log_loss(testY, y_prob)
    print("Loss", loss)


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def run_test_harness_linear_regression(X_train, y_train, X_test, y_test):
    # Initialize the linear regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)


from sklearn.preprocessing import PolynomialFeatures


def run_test_harness_polynomial_regression(X_train, y_train):
    # Create polynomial features
    degree = 2  # Choose the degree of the polynomial
    poly = PolynomialFeatures(degree=degree)
    print("Fit poly")
    X_poly = poly.fit_transform(X_train)

    print("Split data")
    # Split the transformed data into training and testing sets
    X_poly_train, X_poly_test, y_train, y_test = train_test_split(
        X_poly, y_train, test_size=TEST_SIZE, random_state=42
    )
    print("LinearRegression")
    # Initialize and train the linear regression model
    model = LinearRegression()
    print("fit")
    model.fit(X_poly_train, y_train)

    print("predict")
    # Make predictions on the testing data
    y_pred = model.predict(X_poly_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)


def main():
    training_data_input_folder = "./data/07-cropped-images"

    # Get training dataset
    X_train, y_train, overall = helpers.get_training_dataset(training_data_input_folder, TRAINING_DATASET_FOLDERNAMES)
    # print("Lengths of sublists:", lengths)

    # Use same max sizes for training and test to make sure all arrays are the same size.
    X_train = np.array(helpers.get_padded_array(X_train))
    X_train = helpers.replace_image_nan_with_means(X_train)

    X_train = X_train.reshape(X_train.shape[0], -1)

    # Make sure HIGH is 1 and LOW is 0
    # Create array of 0's and 1's for my dataset
    encoded_Y = [CLASS_MAPPING[label] for label in y_train]
    encoded_Y = np.array(encoded_Y, dtype=int)

    POLYNOMIAL = True
    if POLYNOMIAL:
        run_test_harness_polynomial_regression(X_train, encoded_Y)

    trainX, testX, trainY, testY = train_test_split(X_train, encoded_Y, test_size=TEST_SIZE, random_state=42)
    print(trainX.shape, len(trainY), testX.shape, len(testY))

    # Make all predictions = 1
    train_yhat = np.asarray(np.ones(len(trainY)))
    test_yhat = np.asarray(np.ones(len(testY)))

    # evaluate predictions
    train_score = fbeta_score(trainY, train_yhat, average="binary", beta=0.5)
    test_score = fbeta_score(testY, test_yhat, average="binary", beta=0.5)
    print("All Ones (sklearn): train=%.3f, test=%.3f" % (train_score, test_score))

    # run_test_harness_linear_regression(trainX, trainY, testX, testY)
    # run_test_harness_logistic_regression(trainX, trainY, testX, testY)


if __name__ == "__main__":
    main()
