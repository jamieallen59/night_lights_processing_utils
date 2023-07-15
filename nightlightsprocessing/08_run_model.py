import numpy as np
import rasterio
import os
from . import helpers

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers

TEMP_IGNORE_LIST = ["Lucknow-buffer-1-miles", "Sitapur-buffer-1-miles"]
# TEMP_IGNORE_LIST = ["Lucknow-buffer-1-miles"]

TEST_SIZE = 0.4


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


def _replace_nan_with_mean(array):
    array[np.isnan(array)] = np.nanmean(array)

    return array


# ----- Tensorflow keras -----
def run_keras(lights_data_combined, learn_values):
    print("Running keras model")
    # Map the class labels to numeric values
    class_mapping = {"HIGH": 0, "LOW": 1}
    lights_data_combined_classifications = np.array([class_mapping[val] for val in learn_values])

    print("Normalizing data")
    # Normalize the input data
    lights_data_combined = (lights_data_combined - np.nanmin(lights_data_combined)) / (
        np.nanmax(lights_data_combined) - np.nanmin(lights_data_combined)
    )

    lights_data_combined = replace_image_nan_with_means(lights_data_combined)
    # print(lights_data_combined[:4])
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        lights_data_combined, lights_data_combined_classifications, test_size=TEST_SIZE, random_state=0
    )

    print("Defining model architecture")
    # Define the CNN model architecture
    model = keras.Sequential(
        [
            # layers.Reshape(target_shape=(6, 7, 1), input_shape=(6, 7)),  # Reshape for CNN input
            # layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            # layers.MaxPooling2D(pool_size=(2, 2)),
            # layers.Flatten(),
            # layers.Dense(64, activation="relu"),
            # layers.Dense(2, activation="softmax"),  # Two classes: HIGH and LOW
            # --- More complex architecture with 2 convolutional layer ---
            layers.Reshape(target_shape=(6, 7, 1), input_shape=(6, 7)),
            # Reshapes the input data to a specific shape (6x7x1).
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            # Performs convolutional operations with 64 filters and 3x3 kernel size, applying the ReLU activation function.
            layers.MaxPooling2D(pool_size=(2, 2)),
            # Performs max pooling by reducing the spatial dimensions of the data using a 2x2 window.
            layers.Conv2D(128, kernel_size=(2, 2), activation="relu"),
            # Performs additional convolutional operations with 128 filters and 2x2 kernel size, applying the ReLU activation function.
            layers.Flatten(),
            # Flattens the multidimensional data into a 1-dimensional array.
            layers.Dense(128, activation="relu"),
            # Connects all the neurons in the previous layer with 128 neurons, using the ReLU activation function.
            layers.Dropout(0.5),
            # Applies dropout regularization, randomly setting 50% of the input units to 0 during training.
            layers.Dense(2, activation="softmax"),
            # Creates the output layer with 2 neurons, representing the classes, and applies the softmax activation function to obtain class probabilities.
        ]
    )
    print("Model weights", len(model.weights))

    print("Compiling model...")
    # Compile the model
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    print("Training model...")
    # Train the model
    # In general, a larger batch size will require more memory and will take longer to train,
    # but it may also lead to better performance. A smaller batch size will require less
    # memory and will train faster, but it may not perform as well.
    batch_size = 32
    # The number of epochs also affects the performance of the model.
    # A larger number of epochs will allow the model to learn more, but it may also
    # lead to overfitting. A smaller number of epochs will train the model faster,
    # but it may not learn as much.
    epochs = 10
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

    print("Evaluating model...")
    # Evaluate the model on test data
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


def main():
    input_folder = "./data/07-cropped-images"

    training_data, training_data_classifications = _get_training_dataset(input_folder)

    max_size_1st = max(arr.shape[0] for arr in training_data)
    max_size_2nd = max(arr.shape[1] for arr in training_data)

    lights_data_reshaped = np.array(get_padded_array(max_size_1st, max_size_2nd, training_data))

    run_keras(lights_data_reshaped, training_data_classifications)
    # run_pytorch(lights_data_combined, learn_values)

    # ----- TENSOR FLOW -----


if __name__ == "__main__":
    main()
