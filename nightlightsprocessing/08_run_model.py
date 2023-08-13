import numpy as np
import rasterio

import tensorflow as tf
import pandas as pd
import sys
import os
import random
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasClassifier
from sklearn.utils import class_weight
from . import helpers
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score
from matplotlib import pyplot
from mlxtend.preprocessing import minmax_scaling
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression

# Currently ALL AT 40% NAN
TRAINING_DATASET_FOLDERNAMES = [
    "Bahraich-buffer-1-miles",
    "Barabanki-buffer-1-miles",
    "Kanpur-buffer-1-miles",
    "Sitapur-buffer-1-miles",
    "Varanasi-buffer-1-miles",
]

REAL_TEST_LOCATION = "Varanasi-buffer-1-miles"
# REAL_TEST_LOCATION = "Barabanki-buffer-1-miles"
TEST_SIZE = 0.3
TUNING = False
EVALUATING = False
PREDICTING = True
CLASS_MAPPING = {"HIGH": 1, "LOW": 0}


# CURRENTLY TRY JUST 300 OF EACH LOCATION
# --- Get training/test datasets ---
def _get_training_dataset(input_folder, training_dataset_foldernames):
    low = []
    high = []

    training_data = []
    training_data_classifications = []

    for directory in training_dataset_foldernames:
        files_path = f"{input_folder}/{directory}"
        print("files_path HIGH", files_path)
        filter = "HIGH.tif"
        lights_on_data_filepaths = helpers.getAllFilesFromFolderWithFilename(files_path, filter)
        for filename in lights_on_data_filepaths:
            filepath = f"{files_path}/{filename}"

            with rasterio.open(filepath) as src:
                array = src.read()
                array = array[0]
                training_data.append(array)
                high.append(array)
                training_data_classifications.append("HIGH")

        print("files_path LOW", files_path)
        filter = "LOW.tif"
        lights_off_data_filepaths = helpers.getAllFilesFromFolderWithFilename(files_path, filter)
        for filename in lights_off_data_filepaths:
            filepath = f"{files_path}/{filename}"

            with rasterio.open(filepath) as src:
                array = src.read()
                array = array[0]
                training_data.append(array)
                low.append(array)
                training_data_classifications.append("LOW")
    print("Training set total size", len(training_data))
    print("HIGH items", training_data_classifications.count("HIGH"))
    print("LOW items", training_data_classifications.count("LOW"))

    return training_data, training_data_classifications
    # return training_data, training_data_classifications, low, high


def _get_test_dataset(input_path):
    test_data = []
    test_data_classifications = []

    filter = "HIGH.tif"
    lights_on_data_filepaths = helpers.getAllFilesFromFolderWithFilename(input_path, filter)
    for filename in lights_on_data_filepaths:
        filepath = f"{input_path}/{filename}"

        with rasterio.open(filepath) as src:
            array = src.read()
            # array = _replace_nan_with_mean(array)
            test_data.append(array[0])
            test_data_classifications.append("HIGH")

    filter = "LOW.tif"
    lights_off_data_filepaths = helpers.getAllFilesFromFolderWithFilename(input_path, filter)
    for filename in lights_off_data_filepaths:
        filepath = f"{input_path}/{filename}"

        with rasterio.open(filepath) as src:
            array = src.read()
            # array = _replace_nan_with_mean(array)
            test_data.append(array[0])
            test_data_classifications.append("LOW")

    print("Test set total size", len(test_data))
    print("HIGH items", test_data_classifications.count("HIGH"))
    print("LOW items", test_data_classifications.count("LOW"))
    return test_data, test_data_classifications


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


def replace_image_nan_with_means(lights_data_combined):
    # Calculate mean per image
    image_means = np.nanmean(lights_data_combined, axis=(1, 2))  # Calculate mean along the second and third dimensions

    # Perform mean imputation per image
    updated_images = np.copy(
        lights_data_combined
    )  # Create a copy of the original images to store the updated versions

    # Perform mean imputation per image
    for i in range(updated_images.shape[0]):
        image = updated_images[i]
        image_mean = image_means[i]
        image[np.isnan(image)] = image_mean  # Replace NaN with image mean

    return updated_images


def has_all_nan(image):
    return np.all(np.isnan(image))


def normalise_data(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))


def preprocess_data(data, data_classifications):
    data_classifications = np.array([CLASS_MAPPING[val] for val in data_classifications])

    print("Normalizing data")
    data = normalise_data(data)
    print("Replacing NaN values with mean values")
    data = replace_image_nan_with_means(data)

    return data, data_classifications


def reshape_data(X_train, X_test):
    max_size_1st = max(arr.shape[0] for arr in X_train)
    max_size_2nd = max(arr.shape[1] for arr in X_train)
    print(f"All data reshaped to {max_size_1st}, {max_size_2nd}")
    # Use same max sizes for training and test to make sure all arrays are the same size.
    X_train = np.array(get_padded_array(max_size_1st, max_size_2nd, X_train))
    X_test = np.array(get_padded_array(max_size_1st, max_size_2nd, X_test))

    return X_train, X_test


# --- Create model ---
# Define a function that creates your Keras model
def create_model(hidden_layer_dim, learning_rate, model_loss):
    model = tf.keras.Sequential(
        [
            # 1
            # tf.keras.layers.Reshape(target_shape=(6, 7, 1), input_shape=(6, 7)),
            # tf.keras.layers.Conv2D(64, kernel_size=(1, 1), activation="relu", padding="same", input_shape=(6, 7)),
            # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # tf.keras.layers.Conv2D(128, kernel_size=(1, 1), activation="relu", padding="same"),
            # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(hidden_layer_dim),
            # tf.keras.layers.Dense(1, activation="sigmoid"),
            # 2
            # tf.keras.layers.Reshape(target_shape=(6, 7, 1), input_shape=(6, 7)),
            # tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            # tf.keras.layers.MaxPooling2D(2, 2),
            # tf.keras.layers.Conv2D(128, kernel_size=(2, 2), activation="relu"),
            # tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(128, activation="relu"),
            # tf.keras.layers.Dropout(0.5),
            # tf.keras.layers.Dense(1, activation="softmax"),
            # 3
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(6, 7, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(
                1, activation="sigmoid"
            ),  # Output layer with sigmoid activation for binary classification
        ]
    )

    print("Compiling model for evaluation...")
    # SGD
    # optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
    # model.compile(
    #     loss=model_loss,
    #     optimizer=optimizer,
    #     metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalsePositives()],
    # )

    # RMSProps
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    # model.compile(
    #     loss=model_loss,
    #     optimizer=optimizer,
    #     metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalsePositives()],
    # )

    # ADAM
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    model.compile(
        loss=model_loss,
        optimizer=optimizer,
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalsePositives()],
    )

    print("Summary:", model.summary())

    return model


def data_loader():
    # Use training data set for evaluation
    training_data_input_folder = "./data/07-cropped-images"
    X_train, y_train = _get_training_dataset(training_data_input_folder, TRAINING_DATASET_FOLDERNAMES)

    # if EVALUATING | TUNING:
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=TEST_SIZE, random_state=42)
    # else:
    #     # PREDICTING (on random location)
    #     # Get real test/validation data
    #     test_data_input_path = f"{training_data_input_folder}/{REAL_TEST_LOCATION}"
    #     X_test, y_test = _get_test_dataset(test_data_input_path)

    # Randomise the order of training data
    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)

    # Apply class mappings to labels
    y_train = np.array([CLASS_MAPPING[val] for val in y_train])
    y_test = np.array([CLASS_MAPPING[val] for val in y_test])

    return X_train, y_train, X_test, y_test


# SEPARATE OUT EVALUATING AND PREDICTING PROPERLY
# ----- Run model -----
def main():
    # Current hyperparameters
    # In general, a larger batch size will require more memory and will take longer to train,
    # but it may also lead to better performance. A smaller batch size will require less
    # memory and will train faster, but it may not perform as well.
    batch_size = 4
    # The number of epochs also affects the performance of the model.
    # A larger number of epochs will allow the model to learn more, but it may also
    # lead to overfitting. A smaller number of epochs will train the model faster,
    # but it may not learn as much.
    epochs = 10
    # steps_per_epoch = 10
    hidden_layer_dim = 80
    loss = "binary_crossentropy"
    learning_rate = 0.01
    history = None

    if PREDICTING:
        X_train, y_train, X_test, y_test = data_loader()

        # Step 2: Reshape training and test data
        X_train, X_test = reshape_data(X_train, X_test)

        # Step 3 Replace NaN values with mean values for training
        X_train = replace_image_nan_with_means(X_train)
        X_test = replace_image_nan_with_means(X_test)

        # Step 4: Compute scaling parameters from just training data
        mean_train = np.mean(X_train)
        std_train = np.std(X_train)

        # Step 5: Apply scaling to training data
        X_train_scaled = (X_train - mean_train) / std_train
        X_test_scaled = (X_test - mean_train) / std_train

        # Step 6: rectification (make all values positive)
        X_train_scaled_min = np.min(X_train_scaled)
        X_test_scaled_min = np.min(X_test_scaled)

        print("X_train_scaled_min", X_train_scaled_min)
        print("X_test_scaled_min", X_test_scaled_min)
        if X_train_scaled_min <= 0:
            shift_amount = abs(X_train_scaled_min) + 1  # Adding 1 to ensure all values become positive
            X_train_scaled = X_train_scaled + shift_amount
        if X_test_scaled_min <= 0:
            shift_amount = abs(X_test_scaled_min) + 1  # Adding 1 to ensure all values become positive
            X_test_scaled = X_test_scaled + shift_amount

        # Step 7: Divide all values by highest value
        X_train_scaled_max = np.max(X_train_scaled)
        X_test_scaled_max = np.max(X_test_scaled)
        X_train_scaled = X_train_scaled / X_train_scaled_max
        X_test_scaled = X_test_scaled / X_test_scaled_max

        print("X_train_scaled.shape[0]", X_train_scaled.shape[0])
        print("X_test_scaled.shape[0]", X_test_scaled.shape[0])
        print("X_train_scaled", X_train_scaled[:3])
        print("X_test_scaled", X_test_scaled[:3])
        # Step 8:
        X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 6, 7, 1)
        X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 6, 7, 1)

        # Step 8: get class weights
        # class_weights = class_weight.compute_class_weight(
        #     "balanced", classes=[CLASS_MAPPING["LOW"], CLASS_MAPPING["HIGH"]], y=y_train
        # )
        # print("class_weights", class_weights)
        # class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
        # print("class_weights_dict", class_weights_dict)

        # Step 9: printing
        print("X_train_scaled", X_train_scaled[:3])
        print("X_test_scaled", X_test_scaled[:3])

        print("X_train_scaled total", len(X_train_scaled))
        print("y_train total", len(y_train))
        print("y_train HIGH items", (y_train == CLASS_MAPPING["HIGH"]).sum())
        print("y_train LOW items", (y_train == CLASS_MAPPING["LOW"]).sum())

        model = create_model(hidden_layer_dim=hidden_layer_dim, learning_rate=learning_rate, model_loss=loss)

        # history = model.fit(
        #     X_train_scaled,
        #     y_train,
        #     batch_size=batch_size,
        #     epochs=epochs,
        #     verbose=1,
        #     validation_data=(X_test_scaled, y_test),
        #     # class_weight=class_weights_dict,
        # )

        # print("Evaluating model on train_test_split data...")
        # score = model.evaluate(X_test_scaled, y_test, verbose=1)
        # print("Test loss:", score[0])
        # print("Test accuracy:", score[1])

        history = model.fit(
            X_train_scaled,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # class_weight=class_weights_dict,
        )
        predictions = model.predict(X_test_scaled)
        predicted_labels = np.argmax(predictions, axis=1)

        # Validate test
        mean_values = np.nanmean(X_test_scaled, axis=(1, 2))
        mean_values = np.round(mean_values, decimals=1)
        print("Actual mean values to 1DP", mean_values)

        print("Actual labels", y_test)
        print("Actual labels length", len(y_test))
        print("Actual labels LOW count", np.count_nonzero(y_test == CLASS_MAPPING["LOW"]))
        print("Actual labels HIGH count", np.count_nonzero(y_test == CLASS_MAPPING["HIGH"]))

        print("Prediction labels", predicted_labels)
        print("Prediction labels length", len(predicted_labels))
        print("Prediction labels LOW count", np.count_nonzero(predicted_labels == CLASS_MAPPING["LOW"]))
        print("Prediction labels HIGH count", np.count_nonzero(predicted_labels == CLASS_MAPPING["HIGH"]))

        accuracy = np.mean(predicted_labels == y_test)
        print("Accuracy:", accuracy)

    elif EVALUATING:
        # Step 1: Split the data into training and testing sets
        X_train, y_train, X_test, y_test = data_loader()

        # Step 2: Reshape training and test data
        X_train, X_test = reshape_data(X_train, X_test)

        # Step 3 Replace NaN values with mean values for training and test data
        X_train = replace_image_nan_with_means(X_train)
        X_test = replace_image_nan_with_means(X_test)

        # Step 4: Compute scaling parameters from just training data
        mean_train = np.mean(X_train)
        std_train = np.std(X_train)

        # Step 5: Apply scaling to both training and testing data
        X_train_scaled = (X_train - mean_train) / std_train
        X_test_scaled = (X_test - mean_train) / std_train

        # Step 6: get class weights
        class_weights = class_weight.compute_class_weight("balanced", classes=[0, 1], y=y_train)
        print("class_weights", class_weights)

        class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
        print("class_weights_dict", class_weights_dict)

        model = create_model(hidden_layer_dim=hidden_layer_dim, learning_rate=learning_rate, model_loss=loss)
        print("Training model...")
        history = model.fit(
            X_train_scaled,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            # steps_per_epoch=steps_per_epoch,
            verbose=1,
            validation_data=(X_test_scaled, y_test),
            class_weight=class_weights_dict,
        )

        print("Evaluating model on train_test_split data...")
        score = model.evaluate(X_test_scaled, y_test, verbose=1)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    elif TUNING:
        keras_model = KerasClassifier(
            create_model,
            hidden_layer_dim=hidden_layer_dim,
            learning_rate=learning_rate,
            model_loss=loss,
        )

        helpers.hyperparameter_tuning(keras_model, X_train_scaled, y_train)

        # if not TUNING:
        #     # The list of epochs is stored separately from the rest of history.
        #     epochs_history = history.epoch
        #     print("epochs", epochs_history)

        #     # Isolate the error for each epoch.
        #     # hist = pd.DataFrame(history.history)
        #     # print("hist", hist)

        #     # To track the progression of training, we're going to take a snapshot
        #     # of the model's root mean squared error at each epoch.
        #     # binary_accuracy = hist["binary_accuracy"]
        #     # print("binary_accuracy", binary_accuracy)

        #     # Plots my model architechture
        #     # tf.keras.utils.plot_model(model, to_file="model_architecture.png", show_shapes=True)

    # train_loss = history.history["loss"]
    # val_loss = history.history["val_loss"]

    #     # Gather the trained model's weight and bias.
    #     # trained_weight = model.get_weights()[0]
    #     # trained_bias = model.get_weights()[1]
    #     # print("trained_weight", trained_weight)
    #     # print("trained_bias", trained_bias)

    #     # Plotting
    #     # helpers.plot_the_model(y_test, predictions)
    # helpers.plot_the_loss_curve(epochs, train_loss, val_loss)


# ------- Testing -------------
# baseline model
def create_baseline():
    # create model
    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Reshape(target_shape=(6, 7, 1), input_shape=(6, 7)))
    model.add(tf.keras.layers.Dense(60, input_shape=(6, 7), activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    # Compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def test():
    training_data_input_folder = "./data/07-cropped-images"
    X_train, y_train = _get_training_dataset(training_data_input_folder, TRAINING_DATASET_FOLDERNAMES)
    print("X_train[:3]", X_train[:3])

    # Reshape data and pad with nan's
    max_size_1st = max(arr.shape[0] for arr in X_train)
    max_size_2nd = max(arr.shape[1] for arr in X_train)
    print(f"All data reshaped to {max_size_1st}, {max_size_2nd}")
    # Use same max sizes for training and test to make sure all arrays are the same size.
    X_train = np.array(get_padded_array(max_size_1st, max_size_2nd, X_train))
    X_train = replace_image_nan_with_means(X_train)

    print("y_train[:3]", y_train[:3])

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    print("encoded_Y", encoded_Y)

    # evaluate model with standardized dataset
    scaler = StandardScaler()
    num_instances, num_time_steps, num_features = X_train.shape
    # reshape for standardizing
    X_train = np.reshape(X_train, (-1, num_features))
    X_train = scaler.fit_transform(X_train)
    # reshape back to previous shape
    X_train = np.reshape(X_train, (num_instances, num_time_steps, num_features))
    print("X_train.shape", X_train.shape)
    estimators = []
    # estimators.append(("standardize", scaler))
    estimators.append(("mlp", KerasClassifier(model=create_baseline, epochs=100, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)

    print("pipeline", pipeline)
    # kfold = StratifiedKFold(n_splits=10, shuffle=True)
    # print("kfold", kfold)
    results = cross_val_score(pipeline, X_train, encoded_Y, cv=5, shuffle=False)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


# calculate fbeta score for multi-class/label classification
def fbeta(y_true, y_pred, beta=0.5):
    # calculate elements
    tp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)), axis=-1)
    fp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred - y_true, 0, 1)), axis=-1)
    fn = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true - y_pred, 0, 1)), axis=-1)
    # calculate precision
    precision = tp / (tp + fp + tf.keras.backend.epsilon())

    # calculate recall
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    # calculate fbeta, averaged across each class
    bb = beta**2

    fbeta_score = (1 + bb) * (precision * recall) / (bb * precision + recall + tf.keras.backend.epsilon())

    return fbeta_score


# define cnn model
def define_model(in_shape=(6, 7, 1), out_shape=1):
    model = tf.keras.models.Sequential()

    # Simple
    # model.add(
    #     tf.keras.layers.Conv2D(
    #         32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same", input_shape=in_shape
    #     )
    # )
    # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_uniform"))
    # model.add(tf.keras.layers.Dense(64, activation="relu", kernel_initializer="he_uniform", input_shape=in_shape))
    # model.add(tf.keras.layers.Dropout((0.5)))
    model.add(tf.keras.layers.Dense(out_shape, activation="sigmoid", input_shape=in_shape))

    # Complex
    # model.add(
    #     tf.keras.layers.Conv2D(
    #         32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same", input_shape=in_shape
    #     )
    # )
    # model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Dropout((0.2)))
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Dropout((0.2)))
    # model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    # model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    # model.add(tf.keras.layers.Dropout((0.2)))
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_uniform"))
    # model.add(tf.keras.layers.Dense(64, activation="relu", kernel_initializer="he_uniform"))
    # model.add(tf.keras.layers.Dropout((0.5)))
    # model.add(tf.keras.layers.Dense(out_shape, activation="sigmoid"))
    # compile model
    opt = tf.keras.optimizers.legacy.SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[fbeta, "accuracy"])
    return model


# run the test harness for evaluating a model
def run_test_harness(trainX, trainY, testX, testY):
    # Step 4: Compute scaling parameters from just training data
    mean_train = np.mean(trainX)
    std_train = np.std(trainX)

    # Step 5: Apply scaling to training data
    X_train_scaled = (trainX - mean_train) / std_train
    X_test_scaled = (testX - mean_train) / std_train

    # Step 6: rectification (make all values positive)
    X_train_scaled_min = np.min(X_train_scaled)
    X_test_scaled_min = np.min(X_test_scaled)
    if X_train_scaled_min <= 0:
        shift_amount = abs(X_train_scaled_min) + 1  # Adding 1 to ensure all values become positive
        X_train_scaled = X_train_scaled + shift_amount
    if X_test_scaled_min <= 0:
        shift_amount = abs(X_test_scaled_min) + 1  # Adding 1 to ensure all values become positive
        X_test_scaled = X_test_scaled + shift_amount

    # Step 7: Divide all values by highest value
    X_train_scaled_max = np.max(X_train_scaled)
    # X_test_scaled_max = np.max(X_test_scaled)

    reshaped_trainX = np.expand_dims(trainX, axis=-1)  # 'axis=-1' adds the new dimension as the last dimension
    reshaped_testX = np.expand_dims(testX, axis=-1)  # 'axis=-1' adds the new dimension as the last dimension
    print("reshaped_testX.shape", reshaped_testX.shape)

    # create data generator
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True, vertical_flip=True, rotation_range=90
    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    # prepare iterators
    train_it = train_datagen.flow(reshaped_trainX, trainY, batch_size=128)
    test_it = test_datagen.flow(reshaped_testX, testY, batch_size=128)

    # define model
    model = define_model()
    # fit model
    history = model.fit(
        train_it,
        steps_per_epoch=len(train_it),
        validation_data=test_it,
        validation_steps=len(test_it),
        epochs=200,
        verbose=0,
    )

    # predict
    # predictions = model.predict(X_test_scaled)
    # predicted_labels = np.argmax(predictions, axis=1)
    # Validate test
    # mean_values = np.nanmean(X_test_scaled, axis=(1, 2))
    # mean_values = np.round(mean_values, decimals=1)
    # print("Actual mean values to 1DP", mean_values)

    # print("Actual labels", testY)
    # print("Actual labels length", len(testY))
    # print("Actual labels LOW count", np.count_nonzero(testY == CLASS_MAPPING["LOW"]))
    # print("Actual labels HIGH count", np.count_nonzero(testY == CLASS_MAPPING["HIGH"]))

    # print("Prediction labels", predicted_labels)
    # print("Prediction labels length", len(predicted_labels))
    # print("Prediction labels LOW count", np.count_nonzero(predicted_labels == CLASS_MAPPING["LOW"]))
    # print("Prediction labels HIGH count", np.count_nonzero(predicted_labels == CLASS_MAPPING["HIGH"]))

    # accuracy = np.mean(predicted_labels == testY)
    # print("Accuracy:", accuracy)

    # evaluate model
    loss, fbeta, accuracy = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print("> loss=%.3f, fbeta=%.3f, accuracy=%.3f" % (loss, fbeta, accuracy))
    # learning curves
    summarize_diagnostics(history)


# import tensorflow_io as tfio


def test2():
    training_data_input_folder = "./data/07-cropped-images"
    # result = tf.keras.utils.image_dataset_from_directory(f"{training_data_input_folder}/Bahraich-buffer-1-miles")

    X_train, y_train = _get_training_dataset(training_data_input_folder, TRAINING_DATASET_FOLDERNAMES)
    print("_get_training_dataset X_train.shape", len(X_train))
    print("_get_training_dataset y_train len", len(y_train))

    print("X_train[0]", X_train[0])
    fft2 = np.fft.fft(X_train[0])
    print("fft2", fft2)

    pyplot.imshow(fft2, cmap="gray")
    # pyplot.plot(fft2, cmap="gray")
    pyplot.colorbar()
    pyplot.show()

    # Reshape data and pad with nan's
    max_size_1st = max(arr.shape[0] for arr in X_train)
    max_size_2nd = max(arr.shape[1] for arr in X_train)
    print(f"All data reshaped to {max_size_1st}, {max_size_2nd}")
    # Use same max sizes for training and test to make sure all arrays are the same size.
    X_train = np.array(get_padded_array(max_size_1st, max_size_2nd, X_train))
    X_train = replace_image_nan_with_means(X_train)
    print("X_train[:3]", X_train[:3])

    # tf.signal.fft(X_train[0])

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)

    trainX, testX, trainY, testY = train_test_split(X_train, encoded_Y, test_size=0.3, random_state=1)
    print(trainX.shape, len(trainY), testX.shape, len(testY))

    # Make all predictions = 1
    train_yhat = np.asarray(np.ones(len(trainY)))
    test_yhat = np.asarray(np.ones(len(testY)))

    # evaluate predictions
    train_score = fbeta_score(trainY, train_yhat, average="binary", beta=0.5)
    test_score = fbeta_score(testY, test_yhat, average="binary", beta=0.5)
    print("All Ones (sklearn): train=%.3f, test=%.3f" % (train_score, test_score))

    # evaluate predictions with keras
    train_score = fbeta(tf.keras.backend.variable(trainY), tf.keras.backend.variable(train_yhat))
    test_score = fbeta(tf.keras.backend.variable(testY), tf.keras.backend.variable(test_yhat))
    print("All Ones (keras): train=%.3f, test=%.3f" % (train_score, test_score))

    # run_test_harness(trainX, trainY, testX, testY)


# plot diagnostic learning curves
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


# ------ test 3 normalised_per_location ----------


# CURRENTLY TRY JUST 300 OF EACH LOCATION
# --- Get training/test datasets ---
def _get_training_dataset_test_3(input_folder, training_dataset_foldernames):
    low = []
    high = []

    training_data = []
    training_data_classifications = []

    for directory in training_dataset_foldernames:
        sub_directories_path = f"{input_folder}/{directory}"
        location_sub_directories = os.listdir(sub_directories_path)

        for location_sub_directory in location_sub_directories:
            if not location_sub_directory.startswith("."):
                files_path = f"{input_folder}/{directory}/{location_sub_directory}"
                # print("files_path:", files_path)

                filepaths = helpers.getAllFilesFromFolderWithFilename(files_path, "")

                temp = []

                for filename in filepaths:
                    filepath = f"{files_path}/{filename}"

                    with rasterio.open(filepath) as src:
                        array = src.read()
                        array = array[0]

                        image_mean = np.nanmean(array)

                        # array[np.isnan(array)] = image_mean  # Replace NaN with image mean
                        array[np.isnan(array)] = 0  # OR replace Nan with zero?

                        if "LOW" in filepath:
                            item = {"classification": "LOW", "original": array, "mean": image_mean}
                        else:
                            item = {"classification": "HIGH", "original": array, "mean": image_mean}

                        temp.append(item)

                # # Normalise and scale data per location
                all_temp_items = [item["original"] for item in temp]
                all_temp_items = np.array(all_temp_items)
                # print("all_temp_items", all_temp_items[:3])

                # overall_mean = np.nanmean(all_temp_items)
                # overall_std_deviation = np.nanstd(all_temp_items)
                # print("Overall Mean:", overall_mean)
                # print("Overall Standard Deviation:", overall_std_deviation)

                # Add all scaled items and scaled means to temp array
                column_indices = np.arange(all_temp_items.shape[1])
                scaled = minmax_scaling(all_temp_items, columns=column_indices)

                # Add scaled array and means to dataset
                for i in range(len(temp)):
                    scaled_array = scaled[i]
                    scaled_mean = np.nanmean(scaled_array)
                    temp[i]["scaled"] = scaled_array
                    temp[i]["scaled_mean"] = scaled_mean

                # # process arrays with mean and std
                for item in temp:
                    data = item["scaled"]

                    if item["classification"] == "LOW":
                        training_data.append(data)
                        low.append(data)
                        training_data_classifications.append("LOW")
                    else:
                        training_data.append(data)
                        high.append(data)
                        training_data_classifications.append("HIGH")

                # ------ PLOT MEANS AND SCALED. GOOD FOR THESIS. --------
                # fig, ax = pyplot.subplots(1, 2, figsize=(15, 3))
                # all_means = [item["mean"] for item in temp]
                # sns.histplot(all_means, ax=ax[0], kde=True, legend=False)
                # ax[0].set_title("All means (original data)")
                # all_scaled_means = [item["scaled_mean"] for item in temp]
                # sns.histplot(all_scaled_means, ax=ax[1], kde=True, legend=False)
                # ax[1].set_title("All scaled means")
                # pyplot.show()

    print("Training set total size", len(training_data))
    print("HIGH items", training_data_classifications.count("HIGH"))
    print("LOW items", training_data_classifications.count("LOW"))

    return training_data, training_data_classifications


from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc


# run the test harness for evaluating a model
def run_test_harness_3(trainX, trainY, testX, testY):
    # define model
    model = LogisticRegressionCV()
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
    pyplot.figure(figsize=(8, 6))
    pyplot.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    pyplot.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.title("Receiver Operating Characteristic (ROC)")
    pyplot.legend(loc="lower right")
    pyplot.show()

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


def test_3_normalised_per_location():
    training_data_input_folder = "./data/07-cropped-images"

    # Get training dataset
    X_train, y_train = _get_training_dataset_test_3(training_data_input_folder, TRAINING_DATASET_FOLDERNAMES)

    # Reshape data and pad with nan's
    max_size_1st = max(arr.shape[0] for arr in X_train)
    max_size_2nd = max(arr.shape[1] for arr in X_train)
    print(f"All data reshaped to {max_size_1st}, {max_size_2nd}")
    # Use same max sizes for training and test to make sure all arrays are the same size.
    X_train = np.array(get_padded_array(max_size_1st, max_size_2nd, X_train))
    X_train = replace_image_nan_with_zeros(X_train)

    X_train = X_train.reshape(X_train.shape[0], -1)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    print("encoded_Y", encoded_Y)

    # ------- PLOT MEANS FOR WHOLE DATASET. GOOD FOR THESIS. ---------
    # print(X_train.shape)
    # mean_values = np.nanmean(X_train, axis=(1, 2))
    # # Create x values for the plot (assuming one mean value per row)
    # x_values = np.arange(mean_values.shape[0])
    # # Plot the mean values
    # pyplot.figure(figsize=(10, 6))
    # pyplot.plot(x_values, mean_values, marker="o")
    # pyplot.xlabel("Sample Index")
    # pyplot.ylabel("Mean Value")
    # pyplot.title("Mean Values from the 3D Array")
    # pyplot.grid(True)
    # pyplot.show()

    trainX, testX, trainY, testY = train_test_split(X_train, encoded_Y, test_size=0.4, random_state=1)
    print(trainX.shape, len(trainY), testX.shape, len(testY))

    # Make all predictions = 1
    train_yhat = np.asarray(np.ones(len(trainY)))
    test_yhat = np.asarray(np.ones(len(testY)))

    # evaluate predictions
    train_score = fbeta_score(trainY, train_yhat, average="binary", beta=0.5)
    test_score = fbeta_score(testY, test_yhat, average="binary", beta=0.5)
    print("All Ones (sklearn): train=%.3f, test=%.3f" % (train_score, test_score))

    run_test_harness_3(trainX, trainY, testX, testY)


if __name__ == "__main__":
    # main()
    # test()
    # test2()
    test_3_normalised_per_location()
