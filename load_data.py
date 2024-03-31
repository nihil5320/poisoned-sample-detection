import tensorflow as tf
import tensorflow_datasets as tfds

import os

# see https://www.tensorflow.org/guide/data

# get all the jpg files and load them into test, train and validation datasets
def load_datasets(folder_path, image_size=512, batch_size=32):
    """
    Takes a path to a dataset folder and optionally an image size if we want to resize our image (cannot change aspect ratio).
    
    Folder containing dataset should have train, test and validation folders in the root with the desired datasplit. Next level down folder should be the class names then the files beneath that.

    Args:
        folder_path (string): Path to root folder
        image_size (int, optional): Desired image size as a single int, defaults to 512 which will produce 512x512 images

    Returns:
        train_ds, val_ds, test_ds (datasets): returns the compiled datasets based on all files found in the folder specified
    """
    
    train_ds = load_dataset(os.path.join(folder_path,'train','*/*'),image_size)
    val_ds = load_dataset(os.path.join(folder_path,'validation','*/*'),image_size)
    test_ds = load_dataset(os.path.join(folder_path,'test','*/*'),image_size)
    
    return train_ds, val_ds, test_ds


def load_dataset(folder_path, image_size):
    list_ds = tf.data.Dataset.list_files(folder_path)
    built_ds = list_ds.map(
            lambda x: parse_image(x, image_size),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    return built_ds

# Reads an image from a file, decodes it into a dense tensor
# and resizes it if needed to a fixed shape.
def parse_image(filename, image_size):
    label = tf.strings.split(filename, os.sep)[-2]
    # probably want a mapping function at some point but for now original=0, poisoned=1
    encoded_label = tf.cast(label == 'poisoned', tf.int32)
    
    # read and decode the file
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image,channels=3)

    # resize it if we need to
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    if h!=image_size or w!=image_size:
        # for some reason these output float32, whilst the decode uses uint8
        # we'll stick with uint8 to save some memory
        image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
        #image = tf.image.resize(image, (image_size, image_size))
        image = tf.image.convert_image_dtype(image, tf.uint8)

    return image, encoded_label