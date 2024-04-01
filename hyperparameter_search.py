import keras_tuner as kt
import tensorflow as tf

class CustomHyperModel(kt.HyperModel):
    """
        Model builder for support of keras tuner functionality
        see https://www.tensorflow.org/tutorials/keras/keras_tuner / https://keras.io/examples/vision/image_classification_from_scratch/
        
        This needs to be done as a class so we can give it the input shape/class data when initialising, could hardcode this but would suck when testing resizing
    """
    def __init__(self, input_shape, num_classes):
        """
        Need to give it the expected input shape for the dataset and number of classes (should be 2)

        Args:
            input_shape (tensor): input shape, e.g. (512,512,3)
            num_classes (int): number of classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build(self, hp):
        """
        Takes a Keras tuner hyperparameter object and returns a compiled model based on the provided hyperparameters.
        
        Can be called to build the models for further training once a hyperparameter search has completed.

        Args:
            hp (hyperparameter): Keras tuner hyperparameter object

        Returns:
            model: model built to conform to the specification provided in the hyperparameter file
        """
        # first we'll sort out the variables we'll need to build our model
        # learning rate
        hp_lr = hp.Float("learning_rate", min_value=1e-6, max_value=0.1, step=5, sampling="log")
        # data augmentation, this is a bit overkill but I'm interested in seeing how/if this impacts accuracy
        hp_augnorm = hp.Boolean('aug_norm')
        hp_augflip = hp.Boolean('aug_flip')
        hp_augrotate = hp.Float('aug_rotate', min_value=0, max_value=0.3, step=0.05)
        # size of the first conv layer
        hp_conv1 = hp.Int('conv1_size', min_value=64, max_value=192, step=64)
        # do we want to use skip connections for the middle layers?
        hp_residual = hp.Boolean('residual_connections')
        # now how many times do we want to repeat those layers in that middle block?
        hp_midblocks = hp.Int('midblock_repetitions', min_value=1, max_value=3, step=1)
        # now decide how big we want each of these blocks to be
        hp_midblock1 = hp.Int('midblock_1_size', min_value=256, max_value=512, step=128, parent_name='midblock_repetitions', parent_values=[1,2,3])
        hp_midblock2 = hp.Int('midblock_2_size', min_value=256, max_value=768, step=128, parent_name='midblock_repetitions', parent_values=[2,3])
        hp_midblock3 = hp.Int('midblock_3_size', min_value=256, max_value=768, step=128, parent_name='midblock_repetitions', parent_values=[3])
        # size of the final conv layer
        hp_conv2 = hp.Int('final_conv_size', min_value=256, max_value=1280, step=256)
        # lastly we'll just set the dropout
        hp_dropout = hp.Float('dropout', min_value=0, max_value=0.4, step=0.05)
        
        # now we'll chuck all of this at a model constructor
        model = self.model_builder(
            hp_lr,
            hp_augnorm,
            hp_augflip,
            hp_augrotate,
            hp_conv1,
            hp_residual,
            hp_midblocks,
            hp_midblock1,
            hp_midblock2,
            hp_midblock3,
            hp_conv2,
            hp_dropout
        )

        # and compile the model before returning it, still using logits here instead of softmax as we may wish to utilise this later
        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp_lr),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
        )

        return model
    
    def model_builder(self,hp_lr,hp_augnorm,hp_augflip,hp_augrotate,hp_conv1,hp_residual,hp_midblocks,hp_midblock1,hp_midblock2,hp_midblock3,hp_conv2,hp_dropout):
        """Given a list of parameters will build and return an associated model, used during hyperparameter tuning or to recreate the models after the fact.

        Args:
            hp_lr (float): learning rate
            hp_augnorm (bool): flag indicating whether a norm layer is to be included in data augmentation
            hp_augflip (bool): flag indicating whether a random horizontal and vertical flip layer will be included in data augmentation
            hp_augrotate (float): maximum degree of random rotation for augmentation, 0 indicates rotation layer will be excluded
            hp_conv1 (int): size of the first convolutional layer
            hp_residual (bool): flag indicating whether we are to use residual connections, these will go from the end to the start of the "midblocks"
            hp_midblocks (int): This is the number of times we will repeate the middle block of convolutional layers (block consists of two SeparableConv2D with activation and batch norm layers plus a final pooling layer)
            hp_midblock1 (int): number of units in the convolutional layers in the first iteration of the middle block(s)
            hp_midblock2 (int): number of units in the convolutional layers in the second iteration of the middle block(s)
            hp_midblock3 (int): number of units in the convolutional layers in the third iteration of the middle block(s)
            hp_conv2 (int): size of the final convolutional layer
            hp_dropout (float): value for the dropout layer 

        Returns:
            _type_: _description_
        """
        # add some minor augmentation, ideally avoiding anything which might rescale or otherwise impact perturbations
        if (hp_augnorm or hp_augflip or hp_augrotate):
            aug = []
            if hp_augnorm:
                aug.append(tf.keras.layers.Normalization())
            if hp_augflip:
                aug.append(tf.keras.layers.RandomFlip("horizontal_and_vertical"))
            if hp_augrotate:
                aug.append(tf.keras.layers.RandomRotation(hp_augrotate))
            data_augmentation_layers = tf.keras.Sequential(aug)

        # now start building the model
        inputs = tf.keras.Input(shape=self.input_shape)

        # Entry block
        if (hp_augnorm or hp_augflip or hp_augrotate):
            x = data_augmentation_layers(inputs)
            x = tf.keras.layers.Rescaling(1. / 255)(x) # might want to comment this out if we do it in pre
        else:
            x = tf.keras.layers.Rescaling(1. / 255)(inputs) # might want to comment this out if we do it in pre
        x = tf.keras.layers.Conv2D(hp_conv1, 3, strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        # add another pooling layer to help with model size
        # x = tf.keras.layers.MaxPooling2D(hp_kernel2, strides=2, padding="same")(x)

        if hp_residual:
            previous_block_activation = x  # Set aside residual

        # we'll iterate over and recreate this block for the 1-3 times specified by hp_midblocks
        mid_layers = [hp_midblock1, hp_midblock2, hp_midblock3]
        for size in mid_layers[:hp_midblocks]:
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            if hp_residual:
                residual = tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(
                    previous_block_activation
                )
                x = tf.keras.layers.add([x, residual])  # Add back residual
                previous_block_activation = x  # Set aside next residual

        x = tf.keras.layers.SeparableConv2D(hp_conv2, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        if self.num_classes == 2:
            units = 1
        else:
            units = self.num_classes

        # no point adding this if the dropout is 0.
        if hp_dropout:
            x = tf.keras.layers.Dropout(hp_dropout)(x)
        
        # We specify activation=None so as to return logits, dtype being set as we're using mixed precision
        outputs = tf.keras.layers.Dense(units, dtype='float32', activation=None)(x)
        
        model = tf.keras.Model(inputs, outputs)

        return model



