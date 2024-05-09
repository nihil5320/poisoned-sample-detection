import tensorflow as tf
import os

# see https://www.tensorflow.org/guide/data

# get all the jpg files and load them into test, train and validation datasets
def load_datasets(folder_path, class_map, batch_size, image_size=512, crop_and_pad=True):
    """
    Takes a path to a dataset folder and optionally an image size and resizing method if we want to resize our images (cannot change aspect ratio).
    
    Folder containing dataset should have train, test and validation folders in the root with the desired datasplit. Next level down folder should be the class names then the samples in jpg format beneath that.

    Args:
        folder_path (string): Path to root folder, subfolders should be the individual datasets we will load (e.g. test, validation, training)
        class_map (dict): Dict of expected classes where the key is a sequentially increasing integer, this should match the classes in the folders below test/train/validation
        batch_size (int): Desired batch size
        image_size (int, optional): Desired image size as a single int, defaults to 512 which will produce 512x512 images
        crop_and_pad (bool): whether to crop and pad, if false resizes the images

    Returns:
        ds (dict): returns the compiled datasets based on all files found in the folder(s) specified, key will be the name of the root folder
    """
    
    # get the subdirectories first
    dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    # load datasets for all the subdirectories
    ds = {d: load_dataset(os.path.join(folder_path,d,'*/*'), batch_size, image_size, crop_and_pad) for d in dirs}
    
    return ds


def load_dataset(folder_path, batch_size, image_size, crop_and_pad):
    
    # see https://www.tensorflow.org/guide/data_performance
    
    # get a list of all the files in this dataset folder and subdirectories
    list_ds = tf.data.Dataset.list_files(folder_path)
    
    # now we want iterate over that list of files and map them to a new ds
    built_ds = (
        list_ds
        .map(
            lambda x: parse_image(x, image_size, crop_and_pad),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .cache()
        .shuffle(list_ds.cardinality())
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return built_ds

# Reads an image from a file, decodes it into a dense tensor
# and resizes it if needed to a fixed shape.
def parse_image(filename, image_size, crop_and_pad):
    """
    Takes a filepath to a jpg image and a desired size in pixels for the image (width and height will always be equal).
    
    Reads the image from disk and converts to tensor, assigns a label based on the path and applies crop/padding if necessary.

    Args:
        filename (string): Filepath describing the location of the image on disk
        image_size (int): A single integer that describes the desired height and width of the image

    Returns:
        image: tensor of shape (image_size, image_size, 3) containing the image
        encoded_label: label for classification, will be 1 for poisoned samples
    """
    label = tf.strings.split(filename, os.sep)[-2]
    
    # probably want a mapping function at some point but for now original=0, poisoned=1
    encoded_label = tf.cast(label == 'poisoned', tf.int32)
    
    # read and decode the file
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image,channels=3)

    # resize it if the image dimensions differ to those provided by image_size
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    if h!=image_size or w!=image_size:
        if crop_and_pad:
            image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
        else:
            # note resize outputs float32 unless we use nearest neighbour, whilst decode_jpeg uses uint8
            image = tf.image.resize(image, (image_size, image_size), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # we can convert back to tf.uint8 but this significantly degrades image quality
            # image = tf.image.convert_image_dtype(image, tf.uint8)

    return image, encoded_label