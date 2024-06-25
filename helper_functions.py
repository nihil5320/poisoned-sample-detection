import tensorflow as tf
import matplotlib.pyplot as plt

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

def fresh_metrics():
    return [metric() for metric in METRICS]           

# set up a helper function to evaluate models and summarise training
def eval_model(model_name, model, hist, val_ds, test_ds):
    # print the history
    print(f"\n{model_name} training history")
    plot_history(hist,model_name)
    
    # recompile the model with fresh_metrics
    model.compile(metrics=fresh_metrics())
    
    # then evaluate on validation/test data and summarise
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
        dataset (dataset): Dataset from which samples are taken.
        class_map (dict): Dict containing mapping from class IDs to string description.
        grid_shape (tuple, optional): Shape of the grid in which images are displayed. Defaults to (4, 4).
    """
    # create us some subplots    
    fig, axes = plt.subplots(*grid_shape, figsize=(10, 10))
    
    # iterate over the dataset until we've got enough samples
    images = []
    labels = []
    for batch_images, batch_labels in display_ds:
        images.extend(batch_images.numpy())
        labels.extend(batch_labels.numpy())
        if len(images) >= len(axes.flatten()):
            break
        
    # create / populate figure and axes
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i].astype('uint8'))
        ax.set_title(class_map[labels[i]])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def show_incorrect_predictions(model, dataset, class_map, grid_shape=(4,4)):
    """
    Method to take predictions against a dataset and display the resulting images with their predicted and actual classes.
    
    Prioritises displaying incorrect predictions and will alternate between false positives and false negatives. If it exhausts false positives it will display only false negatives and vice versa.
    
    Once all incorrect predictions have been displayed it will begin displaying correct predictions up to the number provided in num_to_display.

    Args:
        model (tensorflow model): Model to be used for prediction.
        dataset (tensorflow dataset): Dataset that predictions will be made against.
        class_map (dict): Dictionary containing the class map for the dataset.
        grid_shape (int): Number of samples to display. Defaults to 16.
    """
    # create us some subplots    
    fig, axes = plt.subplots(*grid_shape, figsize=(10, 10))
    
    # create lists of false positives, false negatives and correct predictions
    false_positives = []
    false_negatives = []
    correct_predictions = []
    # iterate over the dataset to populate these lists
    for image_batch, labels in dataset:
        # make predictions against this specific batch
        predictions = model.predict_on_batch(image_batch)
        # get a numpy array of the images and round the prediction for comparison to the actual class
        images = image_batch.numpy()
        predicted_classes = [0 if p[0] < .5 else 1 for p in predictions.tolist()]
        actual_classes = labels.numpy().tolist()
        # zip each of the lists, figure out which list it needs to go in and append the data as a tuple
        for image, predicted_class, actual_class in zip(images, predicted_classes, actual_classes):
            if predicted_class == actual_class:
                correct_predictions.append((image, predicted_class, actual_class))
            elif class_map[predicted_class] == 'Original':
                false_positives.append((image, predicted_class, actual_class))
            else:
                false_negatives.append((image, predicted_class, actual_class))
        # stop early if we've got enough samples, excluding correct predictions, to draw our grid
        if len(false_positives)+len(false_negatives) >= len(axes.flatten()):
            break
    
    for i, ax in enumerate(axes.flatten()):
        # to alternate we're prioritising false_positives on even numbers, or where we have false_positives but false_negatives is empty
        if false_positives and (i % 2 == 0 or not false_negatives):
            img, predicted_class, actual_class = false_positives.pop()
        # and the inverse for false_negatives
        elif false_negatives and (i % 2 != 0 or not false_positives):
            img, predicted_class, actual_class = false_negatives.pop()
        # lastly we'll display correct predictions
        elif correct_predictions:
            img, predicted_class, actual_class = correct_predictions.pop()
        else:
            print('Exhausted dataset after {i} samples.')
            break
        # once we've decided which list to assign the variables from we can plot the image
        ax.imshow(img)
        # the below code has been taken from TM358: CNN_01_MNIST.ipynb to display correct predictions in green, incorrect in red
        ax.set_title(f"P: {class_map[predicted_class]} (A: {class_map[actual_class]})",
                                    color=("green" if predicted_class == actual_class else "red"))
        ax.axis('off')
    plt.tight_layout()
    plt.show()