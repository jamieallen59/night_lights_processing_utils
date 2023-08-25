import numpy as np

# import rasterio

import sys

# import os
from sklearn.model_selection import train_test_split
from . import helpers

# from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import fbeta_score
from matplotlib import pyplot

# from mlxtend.preprocessing import minmax_scaling
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc

# Currently ALL AT 40% NAN
TRAINING_DATASET_FOLDERNAMES = [
    "Bahraich-buffer-1-miles",
    "Barabanki-buffer-1-miles",
    "Kanpur-buffer-1-miles",
    "Sitapur-buffer-1-miles",
    "Varanasi-buffer-1-miles",
]

REAL_TEST_LOCATION = "Varanasi-buffer-1-miles"
TEST_SIZE = 0.3
TUNING = False
EVALUATING = False
PREDICTING = True
CLASS_MAPPING = {"HIGH": 1, "LOW": 0}


# --- Data processing ---
def get_padded_array(max_size_1st, max_size_2nd, parent_array):
    # Pad arrays with nan values to match the maximum size
    padded_arrays = []

    for arr in parent_array:
        pad_width = (
            (0, max_size_1st - arr.shape[0]),
            (0, max_size_2nd - arr.shape[1]),
        )
        padded_arr = np.pad(arr, pad_width, constant_values=np.nan)
        padded_arrays.append(padded_arr)

    return padded_arrays


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


# --- Get training/test datasets ---
# def _get_training_dataset(input_folder, training_dataset_foldernames):
#     low = []
#     high = []

#     training_data = []
#     training_data_classifications = []

#     for directory in training_dataset_foldernames:
#         sub_directories_path = f"{input_folder}/{directory}"
#         location_sub_directories = os.listdir(sub_directories_path)

#         for location_sub_directory in location_sub_directories:
#             if not location_sub_directory.startswith("."):
#                 files_path = f"{input_folder}/{directory}/{location_sub_directory}"
#                 # print("files_path:", files_path)

#                 filepaths = helpers.getAllFilesFromFolderWithFilename(files_path, "")

#                 temp = []

#                 for filename in filepaths:
#                     filepath = f"{files_path}/{filename}"

#                     with rasterio.open(filepath) as src:
#                         array = src.read()
#                         array = array[0]

#                         image_mean = np.nanmean(array)

#                         # array[np.isnan(array)] = image_mean  # Replace NaN with image mean
#                         array[np.isnan(array)] = 0  # OR replace Nan with zero?

#                         if "LOW" in filepath:
#                             item = {"classification": "LOW", "original": array, "mean": image_mean}
#                         else:
#                             item = {"classification": "HIGH", "original": array, "mean": image_mean}

#                         temp.append(item)

#                 # # Normalise and scale data per location
#                 all_temp_items = [item["original"] for item in temp]
#                 all_temp_items = np.array(all_temp_items)
#                 # print("all_temp_items", all_temp_items[:3])

#                 # overall_mean = np.nanmean(all_temp_items)
#                 # overall_std_deviation = np.nanstd(all_temp_items)
#                 # print("Overall Mean:", overall_mean)
#                 # print("Overall Standard Deviation:", overall_std_deviation)

#                 # Add all scaled items and scaled means to temp array
#                 column_indices = np.arange(all_temp_items.shape[1])
#                 scaled = minmax_scaling(all_temp_items, columns=column_indices)

#                 # Add scaled array and means to dataset
#                 for i in range(len(temp)):
#                     scaled_array = scaled[i]
#                     scaled_mean = np.nanmean(scaled_array)
#                     temp[i]["scaled"] = scaled_array
#                     temp[i]["scaled_mean"] = scaled_mean

#                 # # process arrays with mean and std
#                 for item in temp:
#                     data = item["scaled"]

#                     if item["classification"] == "LOW":
#                         training_data.append(data)
#                         low.append(data)
#                         training_data_classifications.append("LOW")
#                     else:
#                         training_data.append(data)
#                         high.append(data)
#                         training_data_classifications.append("HIGH")

#                 # ------ PLOT MEANS AND SCALED. GOOD FOR THESIS. --------
#                 # fig, ax = pyplot.subplots(1, 2, figsize=(15, 3))
#                 # all_means = [item["mean"] for item in temp]
#                 # sns.histplot(all_means, ax=ax[0], kde=True, legend=False)
#                 # ax[0].set_title("All means (original data)")
#                 # all_scaled_means = [item["scaled_mean"] for item in temp]
#                 # sns.histplot(all_scaled_means, ax=ax[1], kde=True, legend=False)
#                 # ax[1].set_title("All scaled means")
#                 # pyplot.show()

#     print("Training set total size", len(training_data))
#     print("HIGH items", training_data_classifications.count("HIGH"))
#     print("LOW items", training_data_classifications.count("LOW"))

#     return training_data, training_data_classifications


# run the test harness for evaluating a model
def run_test_harness(trainX, trainY, testX, testY):
    # define model
    # model = LogisticRegressionCV(scoring=["accuracy", "neg_log_loss", "roc_auc"])
    model = LogisticRegressionCV(scoring="accuracy", solver="liblinear")
    # fit model
    model.fit(trainX, trainY)

    # predict
    predictions = model.predict(testX)
    accuracy = accuracy_score(testY, predictions)
    print("Accuracy:", accuracy)

    # Validate test
    mean_values = np.nanmean(testX, axis=0)
    mean_values = np.round(mean_values, decimals=1)

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
    # pyplot.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    # pyplot.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    # pyplot.xlim([0.0, 1.0])
    # pyplot.ylim([0.0, 1.05])
    # pyplot.xlabel("False Positive Rate")
    # pyplot.ylabel("True Positive Rate")
    # pyplot.title("Receiver Operating Characteristic (ROC)")
    # pyplot.legend(loc="lower right")
    # pyplot.show()

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(testY, predictions)

    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate and print ROC AUC score
    print("ROC AUC Score:", roc_auc_score(testY, y_prob))

    # What's the loss
    loss = log_loss(testY, y_prob)
    print("Loss", loss)


def replace_image_nan_with_zeros(lights_data_combined):
    # Perform mean imputation per image
    updated_images = np.copy(
        lights_data_combined
    )  # Create a copy of the original images to store the updated versions

    # Perform mean imputation per image
    for i in range(updated_images.shape[0]):
        image = updated_images[i]
        image[np.isnan(image)] = 0  # Replace NaN with image mean

    return updated_images


def main():
    training_data_input_folder = "./data/07-cropped-images"

    # Get training dataset
    X_train, y_train, overall = helpers.get_training_dataset(training_data_input_folder, TRAINING_DATASET_FOLDERNAMES)

    # Reshape data and pad with nan's
    max_size_1st = max(arr.shape[0] for arr in X_train)
    max_size_2nd = max(arr.shape[1] for arr in X_train)
    print(f"All data reshaped to {max_size_1st}, {max_size_2nd}")
    # Use same max sizes for training and test to make sure all arrays are the same size.
    X_train = np.array(get_padded_array(max_size_1st, max_size_2nd, X_train))
    X_train = replace_image_nan_with_zeros(X_train)

    X_train = X_train.reshape(X_train.shape[0], -1)

    # Make sure HIGH is 1 and LOW is 0
    # Create array of 0's and 1's for my dataset
    encoded_Y = [CLASS_MAPPING[label] for label in y_train]
    encoded_Y = np.array(encoded_Y, dtype=int)

    trainX, testX, trainY, testY = train_test_split(X_train, encoded_Y, test_size=TEST_SIZE, random_state=1)
    print(trainX.shape, len(trainY), testX.shape, len(testY))

    # Make all predictions = 1
    train_yhat = np.asarray(np.ones(len(trainY)))
    test_yhat = np.asarray(np.ones(len(testY)))

    # evaluate predictions
    train_score = fbeta_score(trainY, train_yhat, average="binary", beta=0.5)
    test_score = fbeta_score(testY, test_yhat, average="binary", beta=0.5)
    print("All Ones (sklearn): train=%.3f, test=%.3f" % (train_score, test_score))

    run_test_harness(trainX, trainY, testX, testY)


if __name__ == "__main__":
    main()
