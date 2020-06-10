import seaborn as sns
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA


def pca_decomposition(data):
    pca = PCA(n_components=2)
    return pca.fit_transform(data)


def scale_features(iris_dataset):
    features = iris_dataset.columns.to_list()
    features.pop(features.index("species"))
    scaler = MinMaxScaler()
    scaler.fit(iris_dataset[features])
    return pd.DataFrame(scaler.transform(iris_dataset[features]), columns=features).to_numpy()


if __name__ == '__main__':
    iris = sns.load_dataset("iris")
    x_train = scale_features(iris_dataset=iris)

    input_layer = tf.keras.layers.Dense(20, input_shape=(4,))
    encoder_output = tf.keras.layers.Dense(3)
    decoder_input = tf.keras.layers.Dense(20)
    decoder_output = tf.keras.layers.Dense(4)

    autoencoder = tf.keras.Sequential()
    autoencoder.add(layer=input_layer)
    autoencoder.add(layer=encoder_output)
    autoencoder.add(layer=decoder_input)
    autoencoder.add(layer=decoder_output)
    autoencoder.compile(optimizer="adagrad", loss="mse")

    encoder = tf.keras.Sequential()
    encoder.add(input_layer)
    encoder.add(encoder_output)
    encoder.compile(optimizer="adagrad", loss="mse")

    decoder = tf.keras.Sequential()
    decoder.add(decoder_input)
    decoder.add(decoder_output)
    decoder.compile(optimizer="adagrad", loss="mse")

    autoencoder.fit(x=x_train, y=x_train, epochs=500, verbose=True, batch_size=8)

    encoded_data = encoder.predict(x_train)
    decoded_data = decoder.predict(encoded_data)

    loss = mean_squared_error(y_true=x_train, y_pred=decoded_data)
    print(loss)

    encoded_df = pd.DataFrame(data=encoded_data, columns=["f1", "f2", "f3"])
    encoded_df["species"] = iris["species"]

    pca_decomp = pd.DataFrame(data=pca_decomposition(iris[features]), columns=["pc1", "pc2"])
    pca_decomp["species"] = iris["species"]

    ax = sns.scatterplot(x="f1", y="f2", hue="species", data=encoded_df)
    ax = sns.scatterplot(x="f1", y="f3", style="species", data=encoded_df, ax=ax)
    plt.show()