import keras_tuner as kt
import tensorflow as tf
import gc

class CustomHyperModel(kt.HyperModel):
    """
        Model builder for support of keras tuner functionality
        see https://www.tensorflow.org/tutorials/keras/keras_tuner / https://keras.io/examples/vision/image_classification_from_scratch/
        
        This needs to be done as a class so we can give it the input shape/class data when initialising, could hardcode this but would suck when testing resizing
    """
    def __init__(self, input_shape):
        """
        CustomHyperModel for Keras Tuner, takes expected input shape as argument and implements the build method to return a compiled model based on provided hyperparameters.

        Args:
            input_shape (tensor): input shape, e.g. (512,512,3)
        """
        self.input_shape = input_shape
    
    def build(self, hp):
        """
        Takes a Keras tuner hyperparameter object and returns a compiled model based on the provided hyperparameters.
        
        Can be called to build models for further training once a hyperparameter search has completed.

        Args:
            hp (hyperparameter): Keras tuner hyperparameter object

        Returns:
            model: model built to conform to the specification provided in the hyperparameter file
        """
        # first clearing session data to stop memory usage accruing between trials
        # see: https://github.com/keras-team/keras-tuner/issues/395 / https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
        tf.keras.backend.clear_session()
        gc.collect()
        
        # now we can sort out the variables we'll need to build our model
        params = {
            # learning rate
            "learning_rate": hp.Float("learning_rate", min_value=1e-8, max_value=0.01, step=2, sampling="log"),
            # filters in the first conv layer
            "conv1_filters": hp.Int('conv1_filters', min_value=8, max_value=128, step=2, sampling="log"),
            # do we want to use skip connections for the middle layers?
            "residual_connections": hp.Boolean('residual_connections'),
            # now how many times do we want to repeat those layers in that middle block?
            "block_repetitions": hp.Int('block_repetitions', min_value=1, max_value=3, step=1),
            # now decide how big we want each of these blocks to be
            "block_1_filters": hp.Int('block_1_filters', min_value=64, max_value=576, step=128, parent_name='block_repetitions', parent_values=[1,2,3]),
            "block_2_filters": hp.Int('block_2_filters', min_value=128, max_value=640, step=128, parent_name='block_repetitions', parent_values=[2,3]),
            "block_3_filters": hp.Int('block_3_filters', min_value=256, max_value=768, step=128, parent_name='block_repetitions', parent_values=[3]),
            # filters in the final conv layer
            "final_conv_filters": hp.Int('final_conv_filters', min_value=512, max_value=1024, step=128),
            # lastly we'll just set the dropout
            "dropout": hp.Float('dropout', min_value=0, max_value=0.4, step=0.05),
        }
        
        # and chuck all of this at the model constructor
        model = self.model_builder(**params)
        
        # we'll output the parameter count just for information purposes
        print(f'Compiled model with {model.count_params()} parameters.')

        return model
    
    def model_builder(self,learning_rate,conv1_filters,residual_connections,block_repetitions,block_1_filters,block_2_filters,block_3_filters,final_conv_filters,dropout):
        """
        Given a list of parameters will build and return an associated model, used during hyperparameter tuning or to recreate the models after the fact.

        Args:
            learning_rate (float): learning rate
            conv1_filters (int): number of filters in the first convolutional layer
            residual_connections (bool): flag indicating whether we are to use residual connections, these will go from the end to the start of the "midblocks"
            block_repetitions (int): This is the number of times we will repeate the middle block of convolutional layers (block consists of two SeparableConv2D with activation and batch norm layers plus a final pooling layer)
            block_1_filters (int): number of filters in the convolutional layers in the first iteration of the middle block(s)
            block_2_filters (int): number of filters in the convolutional layers in the second iteration of the middle block(s)
            block_3_filters (int): number of filters in the convolutional layers in the third iteration of the middle block(s)
            final_conv_filters (int): filters in the final convolutional layer
            dropout (float): value for the dropout layer 

        Returns:
            model: compiled model configured as described in the provided arguments
        """
        inputs = tf.keras.Input(shape=self.input_shape)

        x = tf.keras.layers.Rescaling(1. / 255)(inputs) # need to comment this out if we do it in pre
        
        # initial convolutional layer
        x = tf.keras.layers.Conv2D(conv1_filters, 3, strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        if residual_connections:
            previous_block_activation = x  # Set aside residual

        # we'll iterate over and recreate this block for the 1-3 times specified by block_repetitions
        mid_layers = [block_1_filters, block_2_filters, block_3_filters]
        for size in mid_layers[:block_repetitions]:
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            if residual_connections:
                residual = tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(
                    previous_block_activation
                )
                x = tf.keras.layers.add([x, residual])  # Add back residual
                previous_block_activation = x  # Set aside next residual

        x = tf.keras.layers.SeparableConv2D(final_conv_filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        # no point adding this if the dropout is 0.
        if dropout:
            x = tf.keras.layers.Dropout(dropout)(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        # changed this to sigmoid, we're using mixed precision so also specifying dtype else it will be float16
        outputs = tf.keras.layers.Dense(1, dtype='float32', activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs, outputs)

        # and compile the model before returning it
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
        )

        return model



