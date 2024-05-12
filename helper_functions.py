import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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

def show_incorrect_predictions(model, dataset, class_map, num_to_display=16):
    """
    Takes a model and a dataset, makes predictions against the dataset and shows the first 16 samples where the prediction was incorrect.
    
    The samples will be displayed in a grid format with the predicted and actual labels. If num_to_display is higher than the number of incorrect predictions it will begin to display correct predictions.

    Args:
        model (tensorflow model): Model to be used for prediction
        dataset (tensorflow dataset): Dataset that predictions will be made against
        num_to_display (int): Number of incorrect samples to display
    """
    # the below code has been adapted from TM358: CNN_01_MNIST.ipynb
    false_positive = []
    false_negative = []
    correct = []
    #test_predictions = model.predict(dataset)    
    #predict_labels = np.argmax(test_predictions, axis=1)
    #test_labels = np.concatenate([y for x, y in dataset], axis=0)
    for image, label in dataset.unbatch():
        image = tf.expand_dims(image,0)
        prediction = model.predict(image, verbose=0)
        score = float(tf.keras.ops.sigmoid(prediction[0][0]))
        print(score)
        print(label.numpy())
        #print(test_predictions)
        #predict_labels = np.argmax(test_predictions, axis=1)
        #print(predict_labels)
        #predict_labels.extend(np.argmax(test_predictions, -1))
        #test_labels.extend(label.numpy())

    #print(predict_labels)
    #print(test_labels)
    #print(test_labels)
    
    """
    # View the true and predicted labels of sample images
    false_positives = []
    false_negatives = []
    correct_predictions = []
    for images, labels in dataset:
        test_predictions = model.predict_on_batch(images)
        predict_labels = np.argmax(test_predictions, axis=1)
        for i in range(len(predict_labels)):
            p_class = predict_labels[i]
            a_class = np.argmax(labels, axis=1)
            print(p_class)
            print(a_class[i])
            if a_class != p_class:
                if class_map[a_class[i]] == 'Original':
                    false_positives.append((images[i].numpy(),p_class,a_class[i]))
                else:
                    false_negatives.append((images[i].numpy(),p_class,a_class[i]))
            else:
                correct_predictions.append((images[i].numpy(),p_class,a_class[i]))
        
    print(len(false_positives))
    print(len(false_negatives))
    print(len(correct_predictions))
    """
    
    """
    for i in range(len(predict_labels)):
        p_class = predict_labels[i]
        a_class = np.argmax(test_labels[i])
        if a_class != p_class:
            if class_map[test_labels[i]] == 'Original':
                false_positives.append((test_imgs[i],p_class,a_class))
            else:
                false_negatives.append((test_imgs[i],p_class,a_class))
        else:
    
    print(len(false_positives))
    print(len(false_negatives))
    print(len(correct_predictions))
    
    plt.figure(figsize=(15,10))
    for i in range(len(false_positives)):
        img = false_positives[0]
        p_class = false_positives[1]
        a_class = false_positives[2]
        plt.subplot(4,4,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img, cmap=plt.cm.binary)
        p_class = predict_labels[i]
        a_class = np.argmax(test_labels[i])
        plt.title(f"P: {class_map[p_class]} (A: {class_map[a_class]})",
                                    color=("green" if p_class == a_class else "red"))
    plt.show()
            correct_predictions.append((test_imgs[i],p_class,a_class))
    """