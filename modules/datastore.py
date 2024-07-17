import tensorflow as tf
import json
import csv
import os

# see https://www.tensorflow.org/guide/data

# get all the jpg files and load them into test, train and validation datasets
def load_datasets(folder_path, batch_size, image_size=512):
    """
    Takes a path to a dataset folder, batch size and optionally an image size and resizing method if we want to resize our images (cannot change aspect ratio).
    
    Folder containing dataset should have train, test and validation folders in the root with the desired datasplit. Next level down folder should be the class names (poisoned or original) then the samples in jpg format beneath that.

    Args:
        folder_path (string): Path to root folder, subfolders should be the individual datasets we will load (e.g. test, validation, training)
        batch_size (int): Desired batch size
        image_size (int, optional): Desired image size as a single int, defaults to 512 which will produce 512x512 images

    Returns:
        ds (dict): returns the compiled datasets based on all files found in the folder(s) specified, key will be the name of the root folder
    """
    
    # get the subdirectories first
    dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    # load datasets for all the subdirectories
    ds = {d: load_dataset(os.path.join(folder_path,d,'*/*'), batch_size, image_size) for d in dirs}
    
    return ds

def load_dataset(folder_path, batch_size, image_size):
    """_summary_

    Args:
        folder_path (_type_): _description_
        batch_size (_type_): _description_
        image_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    # see https://www.tensorflow.org/guide/data_performance
    
    # get a list of all the files in this dataset folder and subdirectories
    list_ds = tf.data.Dataset.list_files(folder_path)
    
    # now we want iterate over that list of files and map them to a new ds
    # we are cropping most large images in the first map so that we can cache without using too much memory
    # in the second map we are applying a random crop
    built_ds = (
        list_ds
        .map(
            lambda x: parse_image(x),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .cache()
        .map(
            lambda x, y: (
                tf.image.random_crop(x, size=(image_size, image_size, 3)),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .shuffle(500)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return built_ds

def parse_image(filepath):
    """
    Takes a filepath to a jpg image. Reads the image from disk, converts to tensor and returns with label.
    
    Image is cropped to 768x768 if it exceeds this resolutions. Label is assigned based on the name of the directory the image is in.

    Args:
        filepath (string): Filepath describing the location of the image on disk

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

    # resizing some of these so they don't take up all our memory when we cache
    max_res=768
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    if h>max_res and w>max_res:
        image = tf.image.random_crop(image, (max_res, max_res, 3))

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
        print(f'Top {num_to_save} results saved to: {filepath}\n')

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