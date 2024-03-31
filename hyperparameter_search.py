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
        
        # first we'll sort out the variables we'll need to build our model
        # learn rate
        hp_lr = hp.Float("learning_rate", min_value=1e-6, max_value=1, step=5, sampling="log")
        # data augmentation, this is a bit overkill but I'm interested in seeing how/if this impacts accuracy
        hp_dataaug = hp.Boolean('dataaug', default=True, parent_name=None, parent_values=None)
        hp_augnorm = hp.Boolean('augnorm', default=False, parent_name='dataaug', parent_values=True)
        hp_augflip = hp.Boolean('augflip', default=False, parent_name='dataaug', parent_values=True)
        hp_augrotate = hp.Float('image_rotation_factor', min_value=0, max_value=1, step=0.2, default=False, parent_name='dataaug', parent_values=True)
        # size of the first conv layer
        hp_conv1 = hp.Int('conv1_units', min_value=128, max_value=512, step=64)
        # kernel size for the same?
        hp_kernel1 = hp.Choice('kernel sizes', [3,5,7,9], ordered=None, default=None, parent_name=None, parent_values=None)
        # do we want to use skip connections for the middle layers?
        hp_residual = hp.Boolean('residual_connections', default=True, parent_name=None, parent_values=None)
        # now how many times do we want to repeat those layers in that middle block?
        hp_midblocks = hp.Int('midblock_repetitions', min_value=1, max_value=3, step=1)
        # now decide how big we want each of these blocks to be
        hp_midblock1 = hp.Int('midblock_1', min_value=256, max_value=768, step=128, parent_name='midblock_repetitions', parent_values=[1,2,3])
        hp_midblock2 = hp.Int('midblock_2', min_value=256, max_value=768, step=128, parent_name='midblock_repetitions', parent_values=[2,3])
        hp_midblock3 = hp.Int('midblock_3', min_value=256, max_value=768, step=128, parent_name='midblock_repetitions', parent_values=3)
        # size of the final conv layer
        hp_conv1 = hp.Int('final_conv_units', min_value=128, max_value=1024, step=128)
        # lastly we'll just set the dropout
        hp_dropout = hp.Float('dropout', min_value=0, max_value=0.5, step=0.1)
        
        # add some minor augmentation, ideally avoiding anything which might rescale or otherwise impact perturbations
        if hp_dataaug:
            aug = []
            if hp_augnorm:
                aug.append(tf.keras.layers.Normalization())
            if hp_augflip:
                aug.append(tf.keras.layers.RandomFlip("horizontal_and_vertical"))
            if hp_augrotate:
                aug.append(tf.keras.layers.RandomRotation(0.05))
            data_augmentation_layers = tf.keras.Sequential(aug)

        # now start building the model
        inputs = tf.keras.Input(shape=self.input_shape)

        # Entry block
        if hp_dataaug:
            x = data_augmentation_layers(inputs)
            x = tf.keras.layers.Rescaling(1.0 / 255)(x) # might want to comment this out if we do it in pre
        else:
            x = tf.keras.layers.Rescaling(1.0 / 255)(inputs) # might want to comment this out if we do it in pre
        x = tf.keras.layers.Conv2D(hp_conv1, hp_kernel1, strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        if hp_residual:
            previous_block_activation = x  # Set aside residual

        # we'll iterate over and recreate this block for the 1-3 times specified above
        mid_layers = [hp_midblock1, hp_midblock2, hp_midblock3]
        for size in mid_layers[:hp_midblocks]:
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

            if hp_residual:
                # Project residual
                residual = tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(
                    previous_block_activation
                )
                x = tf.keras.layers.add([x, residual])  # Add back residual
                previous_block_activation = x  # Set aside next residual

        x = tf.keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        if self.num_classes == 2:
            units = 1
        else:
            units = self.num_classes

        x = tf.keras.layers.Dropout(hp_dropout)(x)
        # We specify activation=None so as to return logits
        outputs = tf.keras.layers.Dense(units, activation=None)(x)
        
        # now compile the model before returning it
        model = tf.keras.Model(inputs, outputs)
        
        model = model.compile(
            optimizer=tf.keras.optimizers.Adam(hp_lr),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
        )

        return model



