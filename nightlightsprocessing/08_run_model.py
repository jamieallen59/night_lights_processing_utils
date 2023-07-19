import numpy as np
import rasterio
import os
import tensorflow as tf
from sklearn.model_selection import GridSearchCV, train_test_split
from scikeras.wrappers import KerasClassifier
from . import helpers


# TEMP_IGNORE_LIST = ["Lucknow-buffer-1-miles", "Sitapur-buffer-1-miles", "Bahraich-buffer-1-miles"]
# TEMP_IGNORE_LIST = ["Lucknow-buffer-1-miles", "Bahraich-buffer-1-miles", "Barabanki-buffer-1-miles"]
REAL_TEST_LOCATION = "Barabanki-buffer-1-miles"
TEMP_IGNORE_LIST = ["Lucknow-buffer-1-miles", REAL_TEST_LOCATION]
TEST_SIZE = 0.3
TUNING = False
PREDICTING = True


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


# SORT OUT. Have this and also _replace_nan_with_mean
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


def _replace_nan_with_mean(array):
    array[np.isnan(array)] = np.nanmean(array)

    return array


COMPLEX_MODEL = [
    # layers.Reshape(target_shape=(6, 7, 1), input_shape=(6, 7)),  # Reshape for CNN input
    # layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    # layers.MaxPooling2D(pool_size=(2, 2)),
    # layers.Flatten(),
    # layers.Dense(64, activation="relu"),
    # layers.Dense(2, activation="softmax"),  # Two classes: HIGH and LOW
    # --- More complex architecture with 2 convolutional layer ---
    tf.keras.layers.Reshape(target_shape=(6, 7, 1), input_shape=(6, 7)),
    # Reshapes the input data to a specific shape (6x7x1).
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    # Performs convolutional operations with 64 filters and 3x3 kernel size, applying the ReLU activation function.
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # Performs max pooling by reducing the spatial dimensions of the data using a 2x2 window.
    tf.keras.layers.Conv2D(128, kernel_size=(2, 2), activation="relu"),
    # Performs additional convolutional operations with 128 filters and 2x2 kernel size, applying the ReLU activation function.
    tf.keras.layers.Flatten(),
    # Flattens the multidimensional data into a 1-dimensional array.
    tf.keras.layers.Dense(128, activation="relu"),
    # Adds constraints to the model's parameters during training (to limit overfitting)
    # tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
    # Connects all the neurons in the previous layer with 128 neurons, using the ReLU activation function.
    tf.keras.layers.Dropout(0.5),
    # Applies dropout regularization, randomly setting 50% of the input units to 0 during training.
    tf.keras.layers.Dense(2, activation="softmax"),
    # Creates the output layer with 2 neurons, representing the classes, and applies the softmax activation function to obtain class probabilities.
]

SIMPLE_MODEL = [
    tf.keras.layers.Reshape(target_shape=(6, 7, 1), input_shape=(6, 7)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax"),
]


# Define a function that creates your Keras model
def create_model(hidden_layer_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Reshape(target_shape=(6, 7, 1), input_shape=(6, 7)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(hidden_layer_dim),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    return model


# hyperparameters
# In general, a larger batch size will require more memory and will take longer to train,
# but it may also lead to better performance. A smaller batch size will require less
# memory and will train faster, but it may not perform as well.
batch_size = 1
# The number of epochs also affects the performance of the model.
# A larger number of epochs will allow the model to learn more, but it may also
# lead to overfitting. A smaller number of epochs will train the model faster,
# but it may not learn as much.
epochs = 1

hidden_layer_dim = 40
loss = "sparse_categorical_crossentropy"
optimizer = "sgd"


def hyperparameter_tuning(X_train, y_train):
    # Wrap your Keras model in a KerasClassifier
    keras_model = KerasClassifier(create_model, loss=loss, hidden_layer_dim=hidden_layer_dim)

    # Define the hyperparameters and their ranges to explore
    # hyperparameters = {
    #     "learning_rate": [0.001, 0.01, 0.1],
    #     "batch_size": [16, 32, 64],
    #     "optimizer": ["adam", "rmsprop", "sgd"],
    # }

    hyperparameters = {
        # "hidden_layer_dim": [50, 100, 200],
        # "batch_size": [16, 32, 64],
        # "loss": ["sparse_categorical_crossentropy"],
        # "optimizer": ["adam", "rmsprop", "sgd"],
        # "optimizer__learning_rate": [0.0001, 0.001, 0.1],
        # Best Hyperparameters: {'batch_size': 16, 'hidden_layer_dim': 50, 'loss': 'sparse_categorical_crossentropy', 'optimizer': 'sgd', 'optimizer__learning_rate': 0.001}
        # "batch_size": [10, 20, 40, 60, 80, 100],
        # "epochs": [10, 50, 100],
        # Best Hyperparameters: {'batch_size': 10, 'epochs': 10}
        # "hidden_layer_dim": [30, 40, 50, 60, 70],
        # "batch_size": [4, 8, 12, 16],
        # "loss": ["sparse_categorical_crossentropy"],
        # "optimizer": ["adam", "rmsprop", "sgd"],
        # "optimizer__learning_rate": [0.0005, 0.001, 0.0015],
        # "epochs": [6, 8, 10, 12],
        # Best Hyperparameters: {'batch_size': 16, 'epochs': 12, 'hidden_layer_dim': 40, 'loss': 'sparse_categorical_crossentropy', 'optimizer': 'sgd', 'optimizer__learning_rate': 0.0005}
        # "batch_size": [14, 18, 20],
        # "epochs": [11, 12, 13, 15],
        # Best Hyperparameters: {'batch_size': 14, 'epochs': 11}
        # "batch_size": [11, 12, 13, 14, 15, 16],
        # "epochs": [9, 10, 11, 12],
        # Best Hyperparameters: {'batch_size': 11, 'epochs': 9}
        # "batch_size": [9, 10, 11, 12, 13],
        # "epochs": [7, 8, 9, 10],
        # Best Hyperparameters: {'batch_size': 9, 'epochs': 7}
        # "hidden_layer_dim": [40],
        # "optimizer": ["sgd"],
        # "loss": ["sparse_categorical_crossentropy"],
        # "batch_size": [9, 10, 11, 12, 13],
        # "epochs": [7, 8, 9, 10],
        # Best Hyperparameters: {'batch_size': 9, 'epochs': 7, 'hidden_layer_dim': 40, 'loss': 'sparse_categorical_crossentropy', 'optimizer': 'sgd'}
        # "hidden_layer_dim": [40],
        # "optimizer": ["sgd"],
        # "loss": ["sparse_categorical_crossentropy"],
        # "batch_size": [7, 8, 9, 10],
        # "epochs": [5, 6, 7, 8, 9],
        # Best Hyperparameters: {'batch_size': 7, 'epochs': 5, 'hidden_layer_dim': 40, 'loss': 'sparse_categorical_crossentropy', 'optimizer': 'sgd'}
        # "hidden_layer_dim": [40],
        # "optimizer": ["sgd"],
        # "loss": ["sparse_categorical_crossentropy"],
        # "batch_size": [5, 6, 7, 8],
        # "epochs": [3, 4, 5, 6],
        # Best Hyperparameters: {'batch_size': 5, 'epochs': 3, 'hidden_layer_dim': 40, 'loss': 'sparse_categorical_crossentropy', 'optimizer': 'sgd'}
        # "hidden_layer_dim": [40],
        # "optimizer": ["sgd"],
        # "loss": ["sparse_categorical_crossentropy"],
        # "batch_size": [3, 4, 5, 6],
        # "epochs": [1, 2, 3, 4],
        # Best Hyperparameters: {'batch_size': 3, 'epochs': 1, 'hidden_layer_dim': 40, 'loss': 'sparse_categorical_crossentropy', 'optimizer': 'sgd'}
        "hidden_layer_dim": [40],
        "optimizer": ["sgd"],
        "loss": ["sparse_categorical_crossentropy"],
        "batch_size": [1, 2, 3, 4],
        "epochs": [1, 2, 3, 4],
        # Best Hyperparameters: {'batch_size': 1, 'epochs': 1, 'hidden_layer_dim': 40, 'loss': 'sparse_categorical_crossentropy', 'optimizer': 'sgd'}
    }

    grid_search = GridSearchCV(keras_model, hyperparameters, cv=5)

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters found
    print("Best Hyperparameters:", grid_search.best_params_)


def preprocess_data(data, data_classifications):
    class_mapping = {"HIGH": 0, "LOW": 1}
    data_classifications = np.array([class_mapping[val] for val in data_classifications])

    print("Normalizing data")
    # Normalize the input data
    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    data = replace_image_nan_with_means(data)

    return data, data_classifications


# ----- Tensorflow keras -----
def run_keras(training_data, training_data_classifications, test_data, test_data_classifications):
    print("Running keras model")

    X_train, y_train = preprocess_data(training_data, training_data_classifications)

    if PREDICTING:
        X_test, y_test = preprocess_data(test_data, test_data_classifications)
    else:
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=TEST_SIZE, random_state=0)
    print("X_train.shape", X_train.shape)
    print("y_train.shape", y_train.shape)

    print("Defining model architecture")
    # Define the CNN model architecture
    model = create_model(hidden_layer_dim=hidden_layer_dim)

    print("Model weights", len(model.weights))

    if TUNING:
        hyperparameter_tuning(X_train, y_train)
    elif PREDICTING:
        print("Compiling model for prediction...")
        model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

        print("Training model...")
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
        # Perform predictions
        predictions = model.predict(X_test)
        print("predictions", predictions[:10])
        # Obtain the predicted class labels
        predicted_labels = np.argmax(predictions, axis=1)

        # Calculate accuracy
        accuracy = np.mean(predicted_labels == y_test)
        print("Accuracy:", accuracy)
    else:
        print("Compiling model for evaluation...")
        model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

        print("Training model...")
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

        print("Evaluating model on train_test_split data...")
        score = model.evaluate(X_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])


def _get_training_dataset(input_folder):
    training_data = []
    training_data_classifications = []

    for directory in os.listdir(input_folder):
        # Ignore e.g. .DS_Store
        if not directory.startswith(".") and directory not in TEMP_IGNORE_LIST:
            files_path = f"{input_folder}/{directory}"
            print("files_path HIGH", files_path)
            filter = "HIGH.tif"
            lights_on_data_filepaths = helpers.getAllFilesFromFolderWithFilename(files_path, filter)
            for filename in lights_on_data_filepaths[:10]:
                filepath = f"{files_path}/{filename}"

                with rasterio.open(filepath) as src:
                    array = src.read()
                    array = _replace_nan_with_mean(array)
                    training_data.append(array[0])
                    training_data_classifications.append("HIGH")

            print("files_path LOW", files_path)
            filter = "LOW.tif"
            lights_off_data_filepaths = helpers.getAllFilesFromFolderWithFilename(files_path, filter)
            for filename in lights_off_data_filepaths:
                filepath = f"{files_path}/{filename}"

                with rasterio.open(filepath) as src:
                    array = src.read()
                    array = _replace_nan_with_mean(array)
                    training_data.append(array[0])
                    training_data_classifications.append("LOW")

    return training_data, training_data_classifications


def _get_test_dataset(input_folder):
    test_data = []
    test_data_classifications = []

    filter = "HIGH.tif"
    lights_on_data_filepaths = helpers.getAllFilesFromFolderWithFilename(input_folder, filter)
    for filename in lights_on_data_filepaths[:10]:
        filepath = f"{input_folder}/{filename}"

        with rasterio.open(filepath) as src:
            array = src.read()
            array = _replace_nan_with_mean(array)
            test_data.append(array[0])
            test_data_classifications.append("HIGH")

    filter = "LOW.tif"
    lights_off_data_filepaths = helpers.getAllFilesFromFolderWithFilename(input_folder, filter)
    for filename in lights_off_data_filepaths:
        filepath = f"{input_folder}/{filename}"

        with rasterio.open(filepath) as src:
            array = src.read()
            array = _replace_nan_with_mean(array)
            test_data.append(array[0])
            test_data_classifications.append("LOW")

    return test_data, test_data_classifications


def main():
    input_folder = "./data/07-cropped-images"

    training_data, training_data_classifications = _get_training_dataset(input_folder)

    test_data, test_data_classifications = _get_test_dataset(f"{input_folder}/{REAL_TEST_LOCATION}")
    print("test_data", test_data[:10])
    print("test_data", test_data_classifications[:10])

    max_size_1st = max(arr.shape[0] for arr in training_data)
    max_size_2nd = max(arr.shape[1] for arr in training_data)

    training_data_reshaped = np.array(get_padded_array(max_size_1st, max_size_2nd, training_data))
    test_data_reshaped = np.array(get_padded_array(max_size_1st, max_size_2nd, test_data))
    print("test_data_reshaped", test_data_reshaped[:10])

    run_keras(training_data_reshaped, training_data_classifications, test_data_reshaped, test_data_classifications)


if __name__ == "__main__":
    main()
