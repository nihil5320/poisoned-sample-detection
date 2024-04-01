import tensorflow as tf
import os

from helper_functions import display_samples

# see https://www.tensorflow.org/guide/data

# get all the jpg files and load them into test, train and validation datasets
def load_datasets(folder_path, class_map, image_size=512, batch_size=None, preview=False):
    """
    Takes a path to a dataset folder and optionally an image size if we want to resize our image (cannot change aspect ratio).
    
    Folder containing dataset should have train, test and validation folders in the root with the desired datasplit. Next level down folder should be the class names then the files beneath that.

    Args:
        folder_path (string): Path to root folder, subfolders should be the individual datasets we will load (e.g. test, validation, training)
        class_map (dict): Dict of expected classes where the key is a sequentially increasing integer, this should match the classes in the folders below test/train/validation
        image_size (int, optional): Desired image size as a single int, defaults to 512 which will produce 512x512 images
        batch_size (int, optional): Desired batch size, if provided
        preview (bool, optional): Flag to indicate whether you'd like to print a selection of one of the datasets as a preview

    Returns:
        ds (dict): returns the compiled datasets based on all files found in the folder(s) specified, key will be the name of the root folder
    """
    
    # get the subdirectories first
    dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    # load datasets for all the subdirectories
    ds = {d: load_dataset(os.path.join(folder_path,d,'*/*'),image_size) for d in dirs}
    
    # generate preview if we've been asked to, only want to do this for one ds so just take the first
    if preview:
        for name, data in ds.items():
            print(f"Generating preview for dataset '{name}' located in '{folder_path}'.")
            display_samples(data,class_map)
            break
    
    # if we've been asked to batch everything...
    if batch_size:
        for d in ds:
            ds[d] = ds[d].batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    
    return ds


def load_dataset(folder_path, image_size):
    
    # first we want to invert the class_map, so we can lookup the expected encoding from the directory name
    #keys,values = class_map.items()
    #map_class = tf.contrib.lookup.HashTable(
    #    tf.contrib.lookup.KeyValueTensorInitializer(values, keys), -1
    #)
    
    # get a list of all the files in this dataset folder and subdirectories
    list_ds = tf.data.Dataset.list_files(folder_path)
    
    # now we want iterate over that list of files and map them to a new ds
    built_ds = list_ds.shuffle(list_ds.cardinality()).map(
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
        # note resize outputs float32 unless we use nearest neighbour, whilst decode_jpeg uses uint8
        image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
        # image = tf.image.resize(image, (image_size, image_size))
        # image = tf.image.convert_image_dtype(image, tf.uint8)

    return image, encoded_label