
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import backend as K

class Autoencoder():
    """
    Autoencoder reperesents a Deep Convolutional autoencoder atchitecture with 
    mirrored encoder and decoder components.
    """

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim) -> None:
        self.input_shape = input_shape # e.g. [28, 28, 1] [width, height, depth]
        self.conv_filters = conv_filters # e.g. [2, 4, 8]
        self.conv_kernels = conv_kernels # [3, 5, 3]
        self.conv_strides = conv_strides # [1, 2, 2]
        self.latent_space_dim = latent_space_dim # 2

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None

        self._build()

    def summary(self):
        self.encoder.summary()

    def _build(self):
        self._build_encoder()
        #self._build_decoder()
        #self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
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
        layer_number = layer_index+1
        conv_layer = keras.layers. Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )

        x = conv_layer(x)
        x = keras.layers.ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = keras.layers.BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x


    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck (Dense layer)."""
        self._shape_before_bottleneck = K.int_shape(x)[1:] # [2, 7, 7, 32]
        
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(self.latent_space_dim, name="encoder_output")(x)
        return x


if __name__  == "__main__":
    autoencoder = Autoencoder(
        input_shape=(28,28,1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3 ),
        conv_strides=(1, 2, 2,1),
        latent_space_dim=2
    )
    autoencoder.summary()