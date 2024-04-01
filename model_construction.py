import tensorflow as tf

# see https://www.tensorflow.org/tutorials/keras/keras_tuner / https://keras.io/examples/vision/image_classification_from_scratch/
def model_builder(input_shape,num_classes,hp_lr,hp_augnorm,hp_augflip,hp_augrotate,hp_conv1,hp_residual,hp_midblocks,hp_midblock1,hp_midblock2,hp_midblock3,hp_conv2,hp_dropout):
    """Given a list of parameters will build and return an associated model, used during hyperparameter tuning or to recreate the models after the fact.

    Args:
        input_shape (tensor): tensor describing the expected input shape for the model
        num_classes (int): number of classes that the model will be expected to identify
        hp_lr (float): learning rate
        hp_augnorm (bool): flag indicating whether a norm layer is to be included in data augmentation
        hp_augflip (bool): flag indicating whether a random horizontal and vertical flip layer will be included in data augmentation
        hp_augrotate (float): maximum degree of random rotation for augmentation, 0 indicates rotation layer will be excluded
        hp_conv1 (int): number of filters in the first convolutional layer
        hp_residual (bool): flag indicating whether we are to use residual connections, these will go from the end to the start of the "midblocks"
        hp_midblocks (int): This is the number of times we will repeate the middle block of convolutional layers (block consists of two SeparableConv2D with activation and batch norm layers plus a final pooling layer)
        hp_midblock1 (int): number of filters in the convolutional layers in the first iteration of the middle block(s)
        hp_midblock2 (int): number of filters in the convolutional layers in the second iteration of the middle block(s)
        hp_midblock3 (int): number of filters in the convolutional layers in the third iteration of the middle block(s)
        hp_conv2 (int): number of filters in the final convolutional layer
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
    inputs = tf.keras.Input(shape=input_shape)

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
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    # no point adding this if the dropout is 0.
    if hp_dropout:
        x = tf.keras.layers.Dropout(hp_dropout)(x)
    
    # We specify activation=None so as to return logits, dtype being set as we're using mixed precision
    outputs = tf.keras.layers.Dense(units, dtype='float32', activation=None)(x)
      
    model = tf.keras.Model(inputs, outputs)

    return model