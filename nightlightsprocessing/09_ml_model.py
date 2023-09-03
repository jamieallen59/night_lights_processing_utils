import sys
import math
import numpy as np
from tensorflow import keras
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, fbeta_score, accuracy_score, roc_curve
from . import helpers
from . import constants

# Currently ALL AT Up to 60% NAN
TRAINING_DATASET_FOLDERNAMES = [
    "Bahraich-buffer-5-miles",
    "Barabanki-buffer-5-miles",
    "Kanpur-buffer-5-miles",
    "Sitapur-buffer-5-miles",
    # "Varanasi-buffer-5-miles",
]


TEST_SIZE = 0.3
EPOCHS = 400
LEARNING_RATE = 0.001
DECAY_LEARNING_RATE = 0.1
DECAY_RATE = DECAY_LEARNING_RATE / EPOCHS
BATCH_SIZE = 64


# calculate fbeta score for multi-class/label classification
def fbeta(y_true, y_pred, beta=0.5):
    # calculate elements
    tp = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)), axis=-1)
    fp = keras.backend.sum(keras.backend.round(keras.backend.clip(y_pred - y_true, 0, 1)), axis=-1)
    fn = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true - y_pred, 0, 1)), axis=-1)
    # calculate precision
    precision = tp / (tp + fp + keras.backend.epsilon())

    # calculate recall
    recall = tp / (tp + fn + keras.backend.epsilon())

    # calculate fbeta, averaged across each class
    bb = beta**2

    fbeta_score = (1 + bb) * (precision * recall) / (bb * precision + recall + keras.backend.epsilon())

    return fbeta_score


# plot diagnostic learning curves
def summarize_diagnostics(history):
    fig, axes = pyplot.subplots(3, 1, figsize=(8, 12))

    if history.history["loss"] and history.history["val_loss"]:
        # plot loss
        axes[0].set_title("Cross Entropy Loss")
        axes[0].plot(history.history["loss"], color=constants.COLOURS["blue"], label="train")
        axes[0].plot(history.history["val_loss"], color=constants.COLOURS["orange"], label="test")

    if history.history["fbeta"] and history.history["val_fbeta"]:
        # plot accuracy
        axes[1].set_title("Fbeta")
        axes[1].plot(history.history["fbeta"], color=constants.COLOURS["blue"], label="train")
        axes[1].plot(history.history["val_fbeta"], color=constants.COLOURS["orange"], label="test")

    if history.history["accuracy"] and history.history["val_accuracy"]:
        # plot accuracy
        axes[2].set_title("accuracy")
        axes[2].plot(history.history["accuracy"], color=constants.COLOURS["blue"], label="train")
        axes[2].plot(history.history["val_accuracy"], color=constants.COLOURS["orange"], label="test")

    # save plot to file
    pyplot.tight_layout()
    filename = sys.argv[0].split("/")[-1]
    pyplot.savefig(filename + "_plot.png")
    pyplot.close()


# define cnn model
def define_simple_model(in_shape=(24, 27, 1), out_shape=1):
    model = keras.models.Sequential()

    # 3 block vgg style architechture
    # VGG block 1
    model.add(
        keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
            activity_regularizer=keras.regularizers.l1(0.0001),
            input_shape=in_shape,
        )
    )
    # model.add(keras.layers.BatchNormalization())
    model.add(
        keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
            activity_regularizer=keras.regularizers.l1(0.0001),
        )
    )
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.GaussianNoise(0.1))
    model.add(keras.layers.Dropout(0.2))
    # VGG block 2
    # model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.MaxPooling2D((2, 2)))
    # model.add(keras.layers.GaussianNoise(0.1))
    # model.add(keras.layers.Dropout(0.3))
    # # VGG block 3
    # model.add(keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    # model.add(keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.MaxPooling2D((2, 2)))
    # model.add(keras.layers.GaussianNoise(0.1))
    # model.add(keras.layers.Dropout(0.4))

    # Output part of the model
    model.add(keras.layers.Flatten())
    model.add(
        keras.layers.Dense(
            32, activation="relu", kernel_initializer="he_uniform", activity_regularizer=keras.regularizers.l1(0.0001)
        )
    )
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.GaussianNoise(0.1))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(out_shape, activation="sigmoid"))

    # compile model

    # Basic SGD
    # opt = keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9, nesterov=False)
    # SGD w/Decay
    # opt = keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9, decay=DECAY_RATE, nesterov=False)

    # ADAM
    # opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # Adam w/DECAY
    # opt = keras.optimizers.Adam(
    #     learning_rate=DECAY_LEARNING_RATE,
    #     decay=DECAY_RATE,
    # )
    # Adam for DROP
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[fbeta, "accuracy"])

    return model


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 200.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


# run the test harness for evaluating a model
def run_test_harness(trainX, trainY, testX, testY):
    print("Train: X=%s, y=%s" % (trainX.shape, trainY.shape))
    print("Test: X=%s, y=%s" % (testX.shape, testY.shape))

    # 'axis=-1' adds the new dimension as the last dimension
    #  This is the 'channel' propety of the image which is grayscale.
    trainX = np.expand_dims(trainX, axis=-1)
    testX = np.expand_dims(testX, axis=-1)
    print("Train: X=%s, y=%s" % (trainX.shape, trainY.shape))
    print("Test: X=%s, y=%s" % (testX.shape, testY.shape))

    # print("Example data", trainX[:3])
    # To plot some images:
    # for i in range(9):
    #     # define subplot
    #     pyplot.subplot(330 + 1 + i)
    #     # plot raw pixel data
    #     pyplot.imshow(trainX[i], cmap="gray")
    #     pyplot.title(constants.INVERSE_CLASS_MAPPING[trainY[i]])  # Assuming trainY contains class indices
    #     # show the figure
    # pyplot.show()

    # define model
    model = define_simple_model()

    # Data augmentation
    # create data generator
    datagen = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
    )
    # prepare iterator
    it_train = datagen.flow(trainX, trainY, batch_size=64)

    # # learning schedule callback
    lrate = keras.callbacks.LearningRateScheduler(step_decay)
    callbacks_list = [lrate]

    # fit model
    steps = int(trainX.shape[0] / BATCH_SIZE)
    history = model.fit(
        it_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        steps_per_epoch=steps,
        validation_data=(testX, testY),
    )

    evaluation = model.evaluate(testX, testY, verbose=0, return_dict=True)
    print("--- Evaluation ---")
    print("Epochs: ", EPOCHS)
    for name, value in evaluation.items():
        print(f"{name}: {value:.4f}")

    # predict
    predictions = model.predict(testX)
    # Produces array e.g.
    # [0.33934796]
    # [0.99781466]
    # [0.9986592 ]
    # So we label < 0.5 as LOW and > 0.5 as HIGH
    predicted_labels = np.where(predictions > 0.5, 1, 0).flatten()

    print("Actual labels LOW count", np.count_nonzero(testY == constants.CLASS_MAPPING["LOW"]))
    print("Actual labels HIGH count", np.count_nonzero(testY == constants.CLASS_MAPPING["HIGH"]))
    print("Prediction labels LOW count", np.count_nonzero(predicted_labels == constants.CLASS_MAPPING["LOW"]))
    print("Prediction labels HIGH count", np.count_nonzero(predicted_labels == constants.CLASS_MAPPING["HIGH"]))

    count_zeros = 0
    count_ones = 0
    for item in predicted_labels:
        if item == 0:
            count_zeros = count_zeros + 1
        if item == 1:
            count_ones = count_ones + 1

    print("Prediction labels counts (zeros, ones)", count_zeros, count_ones)

    # save model
    model.save("non_varanasi_final_model.keras")
    # # learning curves
    summarize_diagnostics(history)


def train_model():
    training_data_input_folder = "./data/07-cropped-images"

    X_train, y_train, overall = helpers.get_training_dataset(training_data_input_folder, TRAINING_DATASET_FOLDERNAMES)
    print("Is data consistent lengths", len(X_train) == len(y_train))
    X_train = np.array(helpers.get_padded_array(X_train))
    X_train = helpers.replace_image_nan_with_means(X_train)

    encoded_Y = helpers.get_y_encoded(y_train)

    trainX, testX, trainY, testY = train_test_split(
        X_train, encoded_Y, test_size=TEST_SIZE, random_state=42, stratify=encoded_Y
    )
    print(trainX.shape, len(trainY), testX.shape, len(testY))

    # Make all predictions = 1
    train_yhat = np.asarray(np.ones(len(trainY)))
    test_yhat = np.asarray(np.ones(len(testY)))

    # evaluate predictions
    train_score = fbeta_score(trainY, train_yhat, average="binary", beta=0.5)
    test_score = fbeta_score(testY, test_yhat, average="binary", beta=0.5)
    print("All Ones (sklearn): train=%.3f, test=%.3f" % (train_score, test_score))

    # evaluate predictions with keras
    train_score = fbeta(keras.backend.variable(trainY), keras.backend.variable(train_yhat))
    test_score = fbeta(keras.backend.variable(testY), keras.backend.variable(test_yhat))
    print("All Ones (keras): train=%.3f, test=%.3f" % (train_score, test_score))

    run_test_harness(trainX, trainY, testX, testY)


import matplotlib.pyplot as plt


def run_prediction():
    training_data_input_folder = "./data/07-cropped-images"
    PREDICTION_DATASET_FOLDERNAMES = [
        "Bahraich-buffer-5-miles",
        "Barabanki-buffer-5-miles",
        "Kanpur-buffer-5-miles",
        "Sitapur-buffer-5-miles",
        "Varanasi-buffer-5-miles",
    ]
    model_filenames = [
        "non_bahraich_final_model",
        "non_barabanki_final_model",
        "non_kanpur_final_model",
        "non_sitapur_final_model",
        "non_varanasi_final_model",
    ]

    # Initialize arrays to store FPR and TPR for each model
    all_fpr = []
    all_tpr = []
    all_names = []

    for prediction_dataset_foldername, model_filename in zip(PREDICTION_DATASET_FOLDERNAMES, model_filenames):
        location = model_filename.replace("non_", "").replace("_final_model", "")
        print(f"--- {location} ---")

        X_test, y_test, overall = helpers.get_training_dataset(
            training_data_input_folder, [prediction_dataset_foldername]
        )
        X_test = np.array(helpers.get_padded_array(X_test))
        X_test = helpers.replace_image_nan_with_means(X_test)

        # load model
        model = keras.models.load_model(f"{model_filename}.keras", custom_objects={"fbeta": fbeta})

        # predict the class
        y_pred = model.predict(X_test)

        helpers.print_results(y_test, y_pred)

        # Compute ROC curve
        encoded_Y = helpers.get_y_encoded(y_test)
        predicted_labels = np.where(y_pred > 0.5, 1, 0).flatten()
        fpr, tpr, _ = roc_curve(encoded_Y, predicted_labels)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        location = model_filename.replace("non_", "").replace("_final_model", "")
        all_names.append(location.capitalize())

    # Plot all ROC curves on a single graph
    plt.figure(figsize=(8, 6))
    for i, (fpr, tpr, location) in enumerate(zip(all_fpr, all_tpr, all_names)):
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{location} (area = %0.2f)" % roc_auc)

    plt.plot([0, 1], [0, 1], color=constants.COLOURS["grey"], lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) - Multiple Models")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    run_prediction()
    # train_model()
