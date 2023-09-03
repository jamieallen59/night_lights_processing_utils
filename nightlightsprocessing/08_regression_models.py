import numpy as np

# import rasterio

import sys

# import os
from sklearn.model_selection import train_test_split
from . import helpers
from . import constants

# from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, fbeta_score, precision_recall_curve
from matplotlib import pyplot

# from mlxtend.preprocessing import minmax_scaling
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import ConfusionMatrixDisplay

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
    pyplot.plot(history.history["loss"], color=constants.COLOURS["blue"], label="train")
    pyplot.plot(history.history["val_loss"], color=constants.COLOURS["orange"], label="test")
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title("Fbeta")
    pyplot.plot(history.history["fbeta"], color=constants.COLOURS["blue"], label="train")
    pyplot.plot(history.history["val_fbeta"], color=constants.COLOURS["orange"], label="test")
    # plot accuracy
    # pyplot.subplot(211)
    # pyplot.title("accuracy")
    # pyplot.plot(history.history["accuracy"], color=constants.COLOURS["blue"], label="train")
    # pyplot.plot(history.history["val_accuracy"], color=constants.COLOURS["orange"], label="test")

    # save plot to file
    filename = sys.argv[0].split("/")[-1]
    pyplot.savefig(filename + "_plot.png")
    pyplot.close()


# def run_test_harness_polynomial_regression(X_train, y_train, X_test, y_test):
#     # Create polynomial features
#     degree = 2  # Choose the degree of the polynomial
#     poly = PolynomialFeatures(degree=degree)
#     print("Fit poly")
#     X_poly_train = poly.fit_transform(X_train)
#     X_poly_test = poly.fit_transform(X_test)

#     print("LinearRegression")
#     # Initialize and train the linear regression model
#     model = LinearRegression()
#     print("fit")
#     model.fit(X_poly_train, y_train)
#     train_accuracy = model.score(X_poly_train, y_train)
#     print(f"Training accuracy: {train_accuracy * 100:.2f}%")

#     print("LENGHTS", X_test.shape, len(y_test))

#     pyplot.scatter(X_test, y_test, label="Actual Data", color=constants.COLOURS["blue"])

#     # Create a range of x-values for the polynomial line
#     x_range = np.linspace(min(X_test), max(X_test), 100)

#     # Generate predictions for the x_range using the model
#     # Assuming your model is named 'model'
#     y_poly = model.predict(poly.fit_transform(x_range.reshape(-1, 1)))

#     # Plot the polynomial line
#     pyplot.plot(x_range, y_poly, label="Polynomial Line", color=constants.COLOURS["red"])

#     # Set labels and legend
#     pyplot.xlabel("X-axis")
#     pyplot.ylabel("Y-axis")
#     pyplot.legend()

#     # Show the plot
#     pyplot.show()

#     print("predict")
#     # Make predictions on the testing data
#     y_pred = model.predict(X_poly_test)

#     helpers.print_predict_accuracy_preencoded(y_test, y_pred)


# In sample
def in_sample():
    training_data_input_folder = "./data/07-cropped-images"

    PREDICTION_DATASET_FOLDERNAMES = [
        "Bahraich-buffer-5-miles",
        "Barabanki-buffer-5-miles",
        "Kanpur-buffer-5-miles",
        "Sitapur-buffer-5-miles",
        "Varanasi-buffer-5-miles",
    ]

    all_fpr = []
    all_tpr = []
    all_names = []

    for prediction_dataset_foldername in PREDICTION_DATASET_FOLDERNAMES:
        location = prediction_dataset_foldername.replace("-buffer-5-miles", "")
        print(f"--- {location} ---")

        # Get training dataset
        X_train, y_train, overall = helpers.get_training_dataset(
            training_data_input_folder, [prediction_dataset_foldername]
        )

        X_train = np.array(helpers.get_padded_array(X_train))
        X_train = helpers.replace_image_nan_with_means(X_train)
        X_train = X_train.reshape(X_train.shape[0], -1)

        y_train = helpers.get_y_encoded(y_train)

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=TEST_SIZE, random_state=42)
        print(X_train.shape, len(y_train), X_test.shape, len(y_test))

        # Make all predictions = 1
        # evaluate predictions
        train_yhat = np.asarray(np.ones(len(y_train)))
        test_yhat = np.asarray(np.ones(len(y_test)))
        train_score = fbeta_score(y_train, train_yhat, average="binary", beta=0.5)
        test_score = fbeta_score(y_test, test_yhat, average="binary", beta=0.5)
        print("All Ones (sklearn): train=%.3f, test=%.3f" % (train_score, test_score))

        # POLYNOMIAL = True
        # if POLYNOMIAL:
        #     run_test_harness_polynomial_regression(X_train, y_train, X_test, y_test)

        # run_test_harness_linear_regression(X_train, y_train, X_test, y_test)

        # --- LINEAR REGRESSION ---
        LINEAR_REGRESSION = False
        if LINEAR_REGRESSION:
            # Initialize the linear regression model
            model = LinearRegression()
            # Train the model on the training data
            model.fit(X_train, y_train)
            # Get the accuracy against the training data
            train_accuracy = model.score(X_train, y_train)
            print(f"Training accuracy: {train_accuracy * 100:.2f}%")

            # Make predictions on the testing data
            y_pred = model.predict(X_test)

            helpers.print_predict_accuracy_preencoded(y_test, y_pred)

        # --- LOGISTIC REGRESSION ---
        LOGISTIC_REGRESSION = True
        if LOGISTIC_REGRESSION:
            # Initialize the logistic regression model
            model = LogisticRegressionCV(scoring="accuracy", solver="liblinear", class_weight="balanced")
            # Train the model on the training data
            model.fit(X_train, y_train)
            # Get the accuracy against the training data
            train_accuracy = model.score(X_train, y_train)
            print(f"Training accuracy: {train_accuracy * 100:.2f}%")
            y_pred = model.predict(X_test)

            helpers.print_predict_accuracy_preencoded(y_test, y_pred)

        predicted_labels = np.where(y_pred > 0.5, 1, 0).flatten()
        fpr, tpr, _ = roc_curve(y_test, predicted_labels)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_names.append(location)

    # Plot all ROC curves on a single graph
    pyplot.figure(figsize=(8, 6))
    for i, (fpr, tpr, location) in enumerate(zip(all_fpr, all_tpr, all_names)):
        roc_auc = auc(fpr, tpr)
        pyplot.plot(fpr, tpr, lw=2, label=f"{location} (area = %0.2f)" % roc_auc)

    pyplot.plot([0, 1], [0, 1], color=constants.COLOURS["grey"], lw=2, linestyle="--")
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.title("Receiver Operating Characteristic (ROC) - Logistic Regression")
    pyplot.legend(loc="lower right")
    pyplot.show()


# Out of sample
def out_of_sample():
    training_data_input_folder = "./data/07-cropped-images"

    PREDICTION_DATASET_FOLDERNAMES = [
        "Bahraich-buffer-5-miles",
        "Barabanki-buffer-5-miles",
        "Kanpur-buffer-5-miles",
        "Sitapur-buffer-5-miles",
        "Varanasi-buffer-5-miles",
    ]

    all_fpr = []
    all_tpr = []
    all_names = []

    for i in range(len(PREDICTION_DATASET_FOLDERNAMES)):
        training_dataset = PREDICTION_DATASET_FOLDERNAMES.copy()
        test_dataset = training_dataset.pop(i)  # Remove the first item from the copy
        print("training_dataset", training_dataset)
        print("index", i)
        location = test_dataset.replace("-buffer-5-miles", "")
        print(f"--- {location} ---")

        # Get training dataset
        X_train, y_train, overall = helpers.get_training_dataset(
            training_data_input_folder,
            training_dataset,
        )
        X_test, y_test, overall = helpers.get_training_dataset(
            training_data_input_folder,
            [test_dataset],
        )
        # Use same max sizes for training and test to make sure all arrays are the same size.
        X_train = np.array(helpers.get_padded_array(X_train))
        X_train = helpers.replace_image_nan_with_means(X_train)
        X_train = X_train.reshape(X_train.shape[0], -1)
        y_train = helpers.get_y_encoded(y_train)

        X_test = np.array(helpers.get_padded_array(X_test))
        X_test = helpers.replace_image_nan_with_means(X_test)
        X_test = X_test.reshape(X_test.shape[0], -1)
        y_test = helpers.get_y_encoded(y_test)

        # --- LINEAR REGRESSION ---
        LINEAR_REGRESSION = False
        if LINEAR_REGRESSION:
            # Initialize the linear regression model
            model = LinearRegression()
            # Train the model on the training data
            model.fit(X_train, y_train)
            # Get the accuracy against the training data
            train_accuracy = model.score(X_train, y_train)
            print(f"Training accuracy: {train_accuracy * 100:.2f}%")
            # Make predictions on the testing data
            y_pred = model.predict(X_test)
            helpers.print_predict_accuracy_preencoded(y_test, y_pred)

        # --- LOGISTIC REGRESSION ---
        LOGISTIC_REGRESSION = True
        if LOGISTIC_REGRESSION:
            # Initialize the logistic regression model
            model = LogisticRegressionCV(scoring="accuracy", solver="liblinear", class_weight="balanced")
            # Train the model on the training data
            model.fit(X_train, y_train)
            # Get the accuracy against the training data
            train_accuracy = model.score(X_train, y_train)
            print(f"Training accuracy: {train_accuracy * 100:.2f}%")
            y_pred = model.predict(X_test)
            helpers.print_predict_accuracy_preencoded(y_test, y_pred)

        predicted_labels = np.where(y_pred > 0.5, 1, 0).flatten()
        fpr, tpr, _ = roc_curve(y_test, predicted_labels)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_names.append(location)

    # Plot all ROC curves on a single graph
    pyplot.figure(figsize=(8, 6))
    for i, (fpr, tpr, location) in enumerate(zip(all_fpr, all_tpr, all_names)):
        roc_auc = auc(fpr, tpr)
        pyplot.plot(fpr, tpr, lw=2, label=f"{location} (area = %0.2f)" % roc_auc)

    pyplot.plot([0, 1], [0, 1], color=constants.COLOURS["grey"], lw=2, linestyle="--")
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.title("Receiver Operating Characteristic (ROC) - Logistic Regression")
    pyplot.legend(loc="lower right")
    pyplot.show()


if __name__ == "__main__":
    in_sample()
    # out_of_sample()
