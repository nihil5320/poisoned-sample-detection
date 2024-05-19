import tensorflow as tf
import json
import csv
import os

# see https://www.tensorflow.org/guide/data

# get all the jpg files and load them into test, train and validation datasets
def load_datasets(folder_path, batch_size, image_size=512, crop_and_pad=True):
    """
    Takes a path to a dataset folder, batch size and optionally an image size and resizing method if we want to resize our images (cannot change aspect ratio).
    
    Folder containing dataset should have train, test and validation folders in the root with the desired datasplit. Next level down folder should be the class names (poisoned or original) then the samples in jpg format beneath that.

    Args:
        folder_path (string): Path to root folder, subfolders should be the individual datasets we will load (e.g. test, validation, training)
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
def parse_image(filepath, image_size, crop_and_pad):
    """
    Takes a filepath to a jpg image and a desired size in pixels for the image (width and height will always be equal).
    
    Reads the image from disk and converts to tensor, assigns a label based on the path and applies crop/padding if necessary.

    Args:
        filepath (string): Filepath describing the location of the image on disk
        image_size (int): A single integer that describes the desired height and width of the image

    Returns:
        image: tensor of shape (image_size, image_size, 3) containing the image
        encoded_label: label for classification, will be 1 for poisoned samples
    """
    label = tf.strings.split(filepath, os.sep)[-2]
    
    # probably want a mapping function at some point but for now original=0, poisoned=1
    encoded_label = tf.cast(label == 'poisoned', tf.int32)
    
    # read and decode the file
    image = tf.io.read_file(filepath)
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

def save_hp_search(tuner,runtime,num_to_save=5,save_folder='results/'):
    """
    Takes a keras tuner object as an argument and will save the top x results to a CSV file.
    
    Filepath will match the project name given to the tuner object, folder by default will be 'results/'.
    
    If updating save location folder must exist!

    Args:
        tuner (tune): keras tuner object
        runtime (int): execution time for the trial in seconds, will not be saved if results for this trial already exist
        num_to_save (int, optional): number of records to save. Defaults to 5
        save_folder (str, optional): folder in which to output csv. Defaults to 'results/search_results'
    """
    # set up the filepath for this trial based on the project name
    filepath=os.path.join(save_folder,'search_results',f'{tuner.project_name}.csv')
    
    # using the below method so we can also include the score
    results = []
    score = None
    for r in tuner.oracle.get_best_trials(num_to_save):
        # include the project name so we can differentiate them when/if we merge these
        x = {"trial": tuner.project_name}
        # we want the hyperparameter values
        x.update(r.hyperparameters.values)
        # and the models score, we'll also save this for the first result
        if not score:
            score = r.score
        x.update({tuner.oracle.objective.name: r.score})
        results.append(x)
    
    # create the file
    with open(filepath, 'w', newline='') as output_file:
        # get a list of all keys first
        headings = [k for k in {k:None for d in results for k in d}]
        # now write out the headings
        dict_writer = csv.DictWriter(output_file, headings)
        dict_writer.writeheader()
        # and iterate over the items in the list
        dict_writer.writerows(results)
        print(f'\nTop {num_to_save} results for {tuner.project_name} saved to: {filepath}')
    
    filepath=os.path.join(save_folder,'hpsearch_comparison.json')
    trial_result = {tuner.project_name: {'score': score, 'runtime_seconds':runtime}}
    # we'll save the overall results as json, need to update this as we go
    if os.path.exists(filepath):
        results = json.load(open(filepath, 'r'))
        if tuner.project_name not in results.keys():
            results.update(trial_result)
            json.dump(results, open(filepath, 'w'), indent="\t")
            print(f'\nResults for {tuner.project_name} added to: {filepath}')
    else:
        json.dump(trial_result, open(filepath, 'w'), indent="\t")    
        # confirm results saved
        print(f'\nTop {num_to_save} results saved to: {filepath}')

def load_hp_searches(load_folder='results/search_results/',metric='val_accuracy'):
    """
    Method to load all CSV files saved via save_hp_search in a specified folder, folder by default will be 'results/search_results'.
    
    Resulting list is sorted such that index 0 will be the most performant model.

    Args:
        load_folder (str, optional): Folder to load saved hyperparameter searches from. Defaults to 'results/search_results/'.
        val_accuracy (str, optional): Metric used to sort the list of hyperparameter searches, should match that used during the search. Defaults to 'val_accuracy'.
    """
    # first get a list of all the files in the given folder
    files = os.listdir(load_folder)
    
    # limit it to csv files
    files = [f for f in files if f.endswith('.csv')]
    
    # now we can create a list of results, iterate over the above list and populate it
    results_list = []
    for f in files:
        filepath=os.path.join(load_folder,f)
        with open(filepath, 'r') as c:
            r = csv.DictReader(c)
            results_list.extend(list(r))
    
    # finally sort the list before we return it
    results_list = sorted(results_list, key=lambda d: d[metric])[::-1]
    
    return results_list