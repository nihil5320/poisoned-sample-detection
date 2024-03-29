import tensorflow as tf

# see https://www.tensorflow.org/tutorials/keras/keras_tuner
def model_builder(input_shape, num_classes):
    
    # add some minor augmentation, avoiding anything which might rescale or otherwise impact perturbations
    # see https://keras.io/examples/vision/image_classification_from_scratch/
    data_augmentation_layers = tf.keras.Sequential([
        #tf.keras.layers.Normalization(),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.05),
    ])

    def data_augmentation(images):
        for layer in data_augmentation_layers:
            images = layer(images)
        return images

    # now start building the model
    inputs = tf.keras.Input(shape=input_shape)

    # Entry block
    x = data_augmentation_layers(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x) # might want to comment this out if we do it in pre
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 768]:
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

    x = tf.keras.layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = tf.keras.layers.Dense(units, activation=None)(x)
    return tf.keras.Model(inputs, outputs)

