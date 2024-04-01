import keras_tuner as kt
import tensorflow as tf

from model_construction import model_builder

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
        # learning rate
        hp_lr = hp.Float("learning_rate", min_value=1e-6, max_value=0.1, step=5, sampling="log")
        # data augmentation, this is a bit overkill but I'm interested in seeing how/if this impacts accuracy
        hp_augnorm = hp.Boolean('aug_norm')
        hp_augflip = hp.Boolean('aug_flip')
        hp_augrotate = hp.Float('aug_rotate', min_value=0, max_value=0.3, step=0.05)
        # size of the first conv layer
        hp_conv1 = hp.Int('conv1_filters', min_value=64, max_value=192, step=64)
        # do we want to use skip connections for the middle layers?
        hp_residual = hp.Boolean('residual_connections')
        # now how many times do we want to repeat those layers in that middle block?
        hp_midblocks = hp.Int('midblock_repetitions', min_value=1, max_value=3, step=1)
        # now decide how big we want each of these blocks to be
        hp_midblock1 = hp.Int('midblock_1_filters', min_value=256, max_value=512, step=128, parent_name='midblock_repetitions', parent_values=[1,2,3])
        hp_midblock2 = hp.Int('midblock_2_filters', min_value=256, max_value=768, step=128, parent_name='midblock_repetitions', parent_values=[2,3])
        hp_midblock3 = hp.Int('midblock_3_filters', min_value=256, max_value=768, step=128, parent_name='midblock_repetitions', parent_values=[3])
        # size of the final conv layer
        hp_conv2 = hp.Int('final_conv_filters', min_value=256, max_value=1280, step=256)
        # lastly we'll just set the dropout
        hp_dropout = hp.Float('dropout', min_value=0, max_value=0.4, step=0.05)
        
        # now we'll chuck all of this at a model constructor
        model = model_builder(
            self.input_shape,
            self.num_classes,
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



