import seaborn as sns
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import click

NUMBER_OF_REDUCED_FEATURES = 2
ENCODER_HIDDEN_LAYER_NODES = 20
DECODER_HIDDEN_LAYER_NODES = 20

OPTIMIZER = "adagrad"
LOSS = "mse"
N_EPOCHS = 50
BATCH_SIZE = 16


def pca_decomposition(data):
    pca = PCA(n_components=NUMBER_OF_REDUCED_FEATURES)
    return pca.fit_transform(data)


def scale_features(features):
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)


def create_models() -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    input_layer = tf.keras.layers.Input(shape=(4,))
    encoder_input = tf.keras.layers.Dense(ENCODER_HIDDEN_LAYER_NODES)
    encoder_output = tf.keras.layers.Dense(NUMBER_OF_REDUCED_FEATURES)
    decoder_input = tf.keras.layers.Dense(DECODER_HIDDEN_LAYER_NODES,)
    decoder_output = tf.keras.layers.Dense(4)

    autoencoder = tf.keras.Sequential()
    autoencoder.add(layer=input_layer)
    autoencoder.add(layer=encoder_input)
    autoencoder.add(layer=encoder_output)
    autoencoder.add(layer=decoder_input)
    autoencoder.add(layer=decoder_output)
    autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS)

    encoder = tf.keras.Sequential()
    encoder.add(input_layer)
    encoder.add(layer=encoder_input)
    encoder.add(layer=encoder_output)
    encoder.compile(optimizer=OPTIMIZER, loss=LOSS)

    decoder = tf.keras.Sequential()
    decoder.add(layer=decoder_input)
    decoder.add(layer=decoder_output)
    decoder.compile(optimizer=OPTIMIZER, loss=LOSS)

    return autoencoder, encoder, decoder


def main():
    iris = sns.load_dataset("iris")
    features = iris.columns.to_list()
    features.remove("species")

    x_train = scale_features(iris[features])

    autoencoder, encoder, decoder = create_models()

    autoencoder.fit(x=x_train, y=x_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE)

    encoded_data = encoder.predict(x_train)
    decoded_data = decoder.predict(encoded_data)

    loss = mean_squared_error(y_true=x_train, y_pred=decoded_data)
    print(f"Reconstruction MSE = {loss}")

    encoded_feature_labels = [f"f{i+1}" for i in range(NUMBER_OF_REDUCED_FEATURES)]

    encoded_df = pd.DataFrame(data=encoded_data, columns=encoded_feature_labels)
    encoded_df["species"] = iris["species"]
    encoded_df["method"] = ["encoder" for _ in range(len(encoded_df))]

    pca_df = pd.DataFrame(data=pca_decomposition(iris[features]), columns=encoded_feature_labels)
    pca_df["species"] = iris["species"]
    pca_df["method"] = ["pca" for _ in range(len(pca_df))]

    reduced_data = pd.concat([encoded_df, pca_df], ignore_index=True)
    reduced_data.to_csv("reduced_data.csv")

    if NUMBER_OF_REDUCED_FEATURES > 1:
        fig = plt.figure(1)
        ax = sns.scatterplot(x=encoded_feature_labels[0],
                             y=encoded_feature_labels[1],
                             hue="species",
                             style="method",
                             data=reduced_data)

        ax.set_title("Comparison of Principal Components\n and Autoencoder Encoded Data")
        plt.show()


if __name__ == '__main__':
    main()
