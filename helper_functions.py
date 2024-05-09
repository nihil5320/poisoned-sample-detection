from IPython.display import HTML, display

import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import os

# set up some more detailed metrics, code from Evaluating_models_with_precision_and_recall.ipynb
METRICS = [
      lambda : tf.keras.metrics.TruePositives(name='tp'),
      lambda : tf.keras.metrics.FalsePositives(name='fp'),
      lambda : tf.keras.metrics.TrueNegatives(name='tn'),
      lambda : tf.keras.metrics.FalseNegatives(name='fn'), 

      lambda : tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      lambda : tf.keras.metrics.Precision(name='precision'),
      lambda : tf.keras.metrics.Recall(name='recall'),
      lambda : tf.keras.metrics.AUC(name='auc'),
]

# set up plotting the training history as a helper function to save repeating it later
def plot_history(history, model_name):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1,len(acc)+1)
    plt.figure()
    plt.plot(epochs, acc, 'ro', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title(f"{model_name} training and validation accuracy")
    plt.legend()  # Automatic detection of elements to be shown in the legend
    plt.xlabel('Epochs')
    plt.xticks(epochs)
    plt.ylabel('Accuracy')
    plt.show()
    
    plt.figure()
    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(f"{model_name} training and validation loss")
    plt.xlabel('Epochs')
    plt.xticks(epochs)
    plt.ylabel('Loss')
    plt.legend()  # Automatic detection of elements to be shown in the legend
    plt.show()

# and the pretty_cm function from earlier in the block
def pretty_cm(cm, class_names):
    ncols = len(class_names) + 2
    result_table  = '<h3>Confusion matrix</h3>\n'
    result_table += '<table border=1>\n'
    result_table += f'<tr><td>&nbsp;</td><td>&nbsp;</td><th colspan={ncols}>Predicted labels</th></tr>\n'
    result_table += '<tr><td>&nbsp;</td><td>&nbsp;</td>'

    for cn in class_names:
        result_table += f'<td><strong>{class_names[cn]}</strong></td>'
    result_table += '</tr>\n'

    result_table += '<tr>\n'
    result_table += f'<th rowspan={ncols}>Actual labels</th>\n'

    for ai, an in enumerate(class_names):
        result_table += '<tr>\n'
        result_table += f'  <td><strong>{class_names[an]}</strong></td>\n'
        for pi, pn in enumerate(class_names):
            result_table += f'  <td>{cm[ai, pi]}</td>\n'
        result_table += '</tr>\n'
    result_table += "</table>"
    # print(result_table)
    display(HTML(result_table))

def fresh_metrics():
    return [metric() for metric in METRICS]           

# set up a helper function to evaluate models and summarise training
def eval_model(model_name, model, hist, val_ds, test_ds):
    # print the history
    print(f"\n{model_name} training history")
    plot_history(hist,model_name)
    
    # recompile the model with fresh_metrics
    model.compile(metrics=fresh_metrics())
    
    # then summarise and evaluate on validation/test data
    print(f"{model_name} summary")
    model.summary()
    print(f"\n{model_name} evaluation on validation data")
    results = model.evaluate(val_ds, return_dict=True, verbose=0)
    [print(f"{metric}: {score}") for metric,score in results.items()] 
    print(f"\n{model_name} evaluation on test data")
    results = model.evaluate(test_ds, return_dict=True, verbose=0)
    [print(f"{metric}: {score}") for metric,score in results.items()] 
    
    # return the results against test data
    return results

def display_samples(display_ds, class_map, grid_shape=(4, 4)):
    """
    Displays images from a dataset in a grid of the specified dimensions
    
    Args:
        dataset (dataset): dataset iterator
        class_map (dict): dict containing mapping from class IDs to string name
        grid_shape (tuple, optional): _description_. Defaults to (4, 4).
    """
    # create us some subplots    
    fig, axes = plt.subplots(*grid_shape, figsize=(10, 10))
    
    # create an iterator over the dataset
    samples = [(image.numpy(), label.numpy()) for image, label in display_ds][:len(axes.flatten())]
    
    # create / populate figure and axes
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i][0].astype('uint8'))
        ax.set_title(class_map[samples[i][1]])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# function to write a list of dicts to CSV, mostly for serialising found HP configurations
def serialise_hpsearch(tuner,num_to_save=5,save_folder='models/search_results/'):
    """
    Takes a keras tuner object as an argument and will save the top x results to a CSV file.
    
    Filename will match the project name given to the tuner object, folder by default will be 'models/search_results'.
    
    If updating save location folder must exist!

    Args:
        tuner (tune): keras tuner object
        num_to_save (int, optional): number of records to save. Defaults to 5.
        save_folder (str, optional): folder in which to output csv. Defaults to 5.
    """
    # set up the filename based on the project name
    filename=os.path.join(save_folder,f'{tuner.project_name}.csv')
    
    # using the below method so we can also include the score
    results = []
    for r in tuner.oracle.get_best_trials(num_to_save):
        # include the file for when/if we merge these
        x = {"trial": filename.split('/')[-1].split('.')[0]}
        # we want the hyperparameter values
        x.update(r.hyperparameters.values)
        # and the models score
        x.update({tuner.oracle.objective.name: r.score})
        results.append(x)
    
    # lastly create the file
    with open(filename, 'w', newline='') as output_file:
        # get a list of all keys first
        headings = [k for k in {k:None for d in results for k in d}]
        # now write out the headings
        dict_writer = csv.DictWriter(output_file, headings)
        dict_writer.writeheader()
        # and iterate over the items in the list
        dict_writer.writerows(results)
    
    # confirm results saved
    print(f'\nTop {num_to_save} results saved to: {filename}')