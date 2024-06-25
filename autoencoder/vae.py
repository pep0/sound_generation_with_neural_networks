import tensorflow as tf
import tensorflow.keras as keras


from tensorflow.keras import backend as K
import numpy as np

import os
import pickle

# Define the KL divergence loss calculation
def kl_divergence_loss(mu, log_variance):
    kl_loss = -0.5 * tf.math.reduce_sum(1 + log_variance - tf.math.square(mu) - tf.math.exp(log_variance), axis=1)
    return kl_loss

class Bottleneck(keras.layers.Layer):

    def __init__(self, latent_space_dim):
        super().__init__()
        self.latent_space_dim = latent_space_dim
        self.mu = keras.layers.Dense(self.latent_space_dim, name="mu")
        self.log_variance = keras.layers.Dense(self.latent_space_dim, name="log_variance")

    def call(self, inputs):
        mu = self.mu(inputs)
        log_variance = self.log_variance(inputs)
        kl_loss = kl_divergence_loss(mu, log_variance)
        self.add_loss(tf.reduce_mean(kl_loss))  # Optionally add KL loss as layer's loss
        return  mu, log_variance

class VAE:
    """
    VAE reperesents a Deep Convolutional variational vae atchitecture with
    mirrored encoder and decoder components.
    """

    def __init__(
        self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim
    ) -> None:
        self.input_shape = input_shape  # e.g. [28, 28, 1] [width, height, depth]
        self.conv_filters = conv_filters  # e.g. [2, 4, 8]
        self.conv_kernels = conv_kernels  # [3, 5, 3]
        self.conv_strides = conv_strides  # [1, 2, 2]
        self.latent_space_dim = latent_space_dim  # 2
        self.reconstruction_loss_weight = 1000000

        self.encoder = None
        self.decoder = None
        self.model = None
        self._model_input = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(optimizer=optimizer, loss=self._calculate_loss_sum )

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(
            x_train, x_train, batch_size=batch_size, epochs=num_epochs, shuffle=True
        )

    def save(self, dir="."):
        self._create_folder_if_it_doesnt_exist(dir)
        self._save_parameters(dir)
        self._save_weights(dir)

    def load_weights(self, weights_file):
        self.model.load_weights(weights_file)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, dir="."):
        parameters_file = os.path.join(dir, "parameters.pkl")
        weights_file = os.path.join(dir, "vae.weights.h5")

        with open(parameters_file, "rb") as f:
            parameters = pickle.load(f)

        vae = VAE(*parameters)
        vae.load_weights(weights_file)
        return vae

    def _calculate_loss_sum(self, y_true, y_pred):
        reconstruction_loss = self._calculate_reconstruction_loss(y_true, y_pred)
        #kl_loss = self._calculate_kl_loss()
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss #+ kl_loss
        return combined_loss

    def _calculate_reconstruction_loss(self, y_true, y_pred):
        error = y_true - y_pred
        reconstruction_loss = tf.keras.ops.mean(tf.keras.ops.square(error), axis=[1,2,3])

        return reconstruction_loss
    
    def _calculate_kl_loss(self):
        kl_loss = -0.5 * tf.reduce_sum(1 + self.log_variance - tf.square(self.mu) - tf.exp(self.log_variance), axis=1)

        #kl_loss = -0.5 * tf.math.reduce_sum(1 + self.log_variance - tf.math.square(self.mu._keras_tensor) - tf.exp(self.log_variance), axis=1)
        return kl_loss
    
    def _create_folder_if_it_doesnt_exist(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def _save_parameters(self, dir):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim,
        ]
        parameters_file = os.path.join(dir, "parameters.pkl")
        with open(parameters_file, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, dir):
        weights_file = os.path.join(dir, "vae.weights.h5")
        self.model.save_weights(weights_file)

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_vae()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = keras.Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return keras.layers.Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        """creates all convolutional blocks in encoder."""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)

        return x

    def _add_conv_layer(self, layer_index, x):
        """Adds a convolutional block to a graph of layers, consiting of
        conv 2d + ReLu + batch normalization.
        """
        layer_number = layer_index + 1
        conv_layer = keras.layers.Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}",
        )

        x = conv_layer(x)
        x = keras.layers.ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = keras.layers.BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck with Gaussian sampling (Dense layer)."""
        self._shape_before_bottleneck = x.shape[1:]  # [2, 7, 7, 32]

        x = keras.layers.Flatten()(x)
        self.mu, self.log_variance = Bottleneck(latent_space_dim=self.latent_space_dim)(x)

        """A lambda layer wrapes a function into the tensorflow graph"""

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args


            epsilon = tf.random.normal(shape=tf.shape(mu), mean=0.0, stddev=1.0)
            sampled_point = mu + tf.math.exp(log_variance / 2) * epsilon
            return sampled_point

        x = keras.layers.Lambda(
            sample_point_from_normal_distribution, name="encoder_output" , output_shape=(self.latent_space_dim,)
        )([self.mu, self.log_variance])

        return x

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return keras.layers.Input(shape=[self.latent_space_dim], name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)  # e.g. [1, 2, 4] -> 8
        dense_layer = keras.layers.Dense(num_neurons, name="decoder_dense")(
            decoder_input
        )
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        reshape_layer = keras.layers.Reshape(self._shape_before_bottleneck)(dense_layer)
        return reshape_layer

    def _add_conv_transpose_layers(self, x):
        """Add conv transpose blocks."""
        # loop through all the conv layers in reverse order and stop at the first layer
        # first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            # [0, 1, 2] -> [2, 1] (ignore first layer, add 1 to range)
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = keras.layers.Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}",
        )
        x = conv_transpose_layer(x)
        x = keras.layers.ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = keras.layers.BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = keras.layers.Conv2DTranspose(
            filters=1,  # [24, 24, 1]
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}",
        )
        x = conv_transpose_layer(x)
        output_layer = keras.layers.Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def _build_vae(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = keras.Model(model_input, model_output, name="vae")


if __name__ == "__main__":
    vae = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2,
    )
    vae.summary()
