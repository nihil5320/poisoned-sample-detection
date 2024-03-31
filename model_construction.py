import tensorflow as tf

# see https://www.tensorflow.org/tutorials/keras/keras_tuner / https://keras.io/examples/vision/image_classification_from_scratch/
def model_builder(input_shape, num_classes):    
    # first we'll sort out the variables we'll need to build our model
    # learn rate
    hp_lr = 1e-06
    # data augmentation, this is a bit overkill but I'm interested in seeing how/if this impacts accuracy
    hp_dataaug = False
    hp_augnorm = False
    hp_augflip = False
    hp_augrotate = False
    # size of the first conv layer
    hp_conv1 = 192
    # kernel size for the same?
    hp_kernel1 = 3
    # do we want to use skip connections for the middle layers?
    # hp_residual = hp.Boolean('residual_connections', default=True, parent_name=None, parent_values=None)
    # now how many times do we want to repeat those layers in that middle block?
    hp_midblocks = 1
    # now decide how big we want each of these blocks to be
    hp_midblock1 = 640
    hp_midblock2 = None
    hp_midblock3 = None
    # lastly we'll just set the dropout
    hp_dropout = 0.4
    
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
    inputs = tf.keras.Input(shape=input_shape)

    # Entry block
    if hp_dataaug:
        x = data_augmentation_layers(inputs)
        x = tf.keras.layers.Rescaling(1.0 / 255)(x) # might want to comment this out if we do it in pre
    else:
        x = tf.keras.layers.Rescaling(1.0 / 255)(inputs) # might want to comment this out if we do it in pre
    x = tf.keras.layers.Conv2D(hp_conv1, hp_kernel1, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

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
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

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

    return tf.keras.Model(inputs, outputs)