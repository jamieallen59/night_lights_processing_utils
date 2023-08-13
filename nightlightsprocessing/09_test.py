import tensorflow as tf

import pandas as pd

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# baseline model
def create_baseline():
    # create model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(60, input_shape=(60,), activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    # Compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def main():
    filepath = "./nightlightsprocessing/sonar.csv"
    # load dataset
    dataframe = pd.read_csv(filepath, header=None)
    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:60]
    print("X.dtype", X.dtype)
    X = dataset[:, 0:60].astype(float)
    print("X.dtype", X.dtype)
    print("X.shape", X.shape)

    Y = dataset[:, 60]
    print("X[:3]", X[:3])
    print("Y[:3]", Y[:3])

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    print("encoded_Y", encoded_Y)

    # evaluate model with standardized dataset
    estimator = KerasClassifier(model=create_baseline, epochs=100, batch_size=5, verbose=0)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


if __name__ == "__main__":
    main()
