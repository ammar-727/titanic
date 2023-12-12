import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import time



# Importing and cleaning data
# importing csv data
titanic_data = pd.read_csv('train.csv')

# Removing unnecessary rows
titanic_useable_data = titanic_data[['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
titanic_useable_data = titanic_useable_data.copy()

# Adding "missing_age" column where value=1 if "age" is given and value=0 if "age" is not given and replacing missing values in the 'Age' column with 0
titanic_useable_data['Age_missing'] = titanic_useable_data['Age'].isna().astype(int)
titanic_useable_data['Age'] = titanic_useable_data['Age'].fillna(0)

# Adding a new column "titles" that consists only title part form the whole name. e.g. "Braund, Mr. Owen Harris" becomes "Mr". Also removing the column "Name"
titanic_useable_data['Titles'] = titanic_useable_data['Name'].str.split(', ').str[1].str.split('.').str[0]
titanic_useable_data = titanic_useable_data.drop('Name', axis=1)

# Using one-hot-encoding foe "Sex", "Embarked", and "Titles"
titanic_useable_data = pd.get_dummies(titanic_useable_data, columns=['Sex', 'Embarked', 'Titles'])



# Splitting Data
# Splitting features and targets
titanic_features = titanic_useable_data.drop('Survived', axis=1)
titanic_target = titanic_useable_data['Survived']

# Normalizing Features
titanic_features_norm = (titanic_features - titanic_features.mean()) / titanic_features.std()

# Converting Pandas DataFrame to numpy array
titanic_features = titanic_features_norm.values
titanic_targets = titanic_target.values

# Splitting data into Train/val/test sets (70%/15%/15%)
train_ratio = 0.7
#val_test_ratio = 0.5
# X_train, X_val_test, y_train, y_val_test = train_test_split(titanic_features, titanic_targets, test_size=(1 - train_ratio), random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=(val_test_ratio), random_state=42)
X_train, X_val, y_train, y_val = train_test_split(titanic_features, titanic_targets, test_size=(1 - train_ratio), random_state=10)



# Custom metric functions
from keras import backend as K

def Recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def Precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def F1Score(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# # Callback to save model weights after each epoch
# class SaveWeightsCallback(tf.keras.callbacks.Callback):
#     def __init__(self, save_dir, naming_convention):
#         super(SaveWeightsCallback, self).__init__()
#         self.save_dir = save_dir
#         self.naming_convention = naming_convention

#     def on_epoch_end(self, epoch, logs=None):
#         model_name = f"{self.naming_convention}_weights_epoch_{epoch + 1}.h5"
#         model_path = os.path.join(self.save_dir, model_name)
#         self.model.save_weights(model_path)



# 2nd batch testing
lrs = [0.0003, 0.0001, 0.0002, 0.0001, 0.0001, 0.0005, 0.0008, 0.01]
momentums = [0.7, 0.7, 0.8, 0.9, 0.9, 0.7, 0.8, 0.95]
batch_sizes = [256, 256, 32, 32, 32, 64, 32, 128]
epochs = [15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000]
activations = ['relu', 'tanh', 'tanh', 'relu', 'tanh', 'tanh', 'relu', 'relu']
lr_decays = [0, 0, 0.7, 0.95, 0.95, 0.95, 0.5, 0.95]
dropout_rates = [0.7, 0.5, 0.75, 0.7, 0.6, 0.75, 0.85, 0.85]
naming_convention = [2, 2, 4, 6, 6, 9, 11, 13]

# one = ['one']
# fignum = 0



for i in range(len(lrs)):

    # Setting hyperparameters
    # Randomly selecting hyperparameter values from manually found ones
    # learningrate = random.choice(lrs)
    # momentum = random.choice(momentums)
    # batchsize = random.choice(batch_sizes)
    # lr_decay = random.choice(lr_decays)
    # dropout_rate = random.choice(dropout_rates)

    # for act in activations:
    # for dor in dropout_rates:
        
    learningrate = lrs[i]
    momentum = momentums[i]
    batchsize = batch_sizes[i]
    lr_decay = lr_decays[i]
    dropout_rate = dropout_rates[i]
    activation = activations[i]
    nmg_conv = naming_convention[i]
    epoch = epochs[i]

    # Create a sequential model
    model = tf.keras.Sequential()

    # Print the selected values
    print("Learning Rate:", learningrate)
    print("Momentum:", momentum)
    print("Batch Size:", batchsize)
    print("Number of Epochs:", epoch)
    print("Learning Rate Decay:", lr_decay)
    print("Activation Function:", activation)
    print("Dropout Rate:", dropout_rate)
    print("Naming Convention:", nmg_conv)

    # Adding dense layers
    model.add(layers.Input(shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Dense(units=32, activation=activation, kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=True, bias_initializer='zeros'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_rate, seed=10))
    model.add(tf.keras.layers.Dense(units=64, activation=activation, kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=True, bias_initializer='zeros'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_rate, seed=10))
    model.add(tf.keras.layers.Dense(units=128, activation=activation, kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=True, bias_initializer='zeros'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_rate, seed=10))
    model.add(tf.keras.layers.Dense(units=256, activation=activation, kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=True, bias_initializer='zeros'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_rate, seed=10))
    model.add(tf.keras.layers.Dense(units=512, activation=activation, kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=True, bias_initializer='zeros'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_rate, seed=10))
    model.add(tf.keras.layers.Dense(units=256, activation=activation, kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=True, bias_initializer='zeros'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_rate, seed=10))
    model.add(tf.keras.layers.Dense(units=128, activation=activation, kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=True, bias_initializer='zeros'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_rate, seed=10))
    model.add(tf.keras.layers.Dense(units=64, activation=activation, kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=True, bias_initializer='zeros'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_rate, seed=10))
    model.add(tf.keras.layers.Dense(units=32, activation=activation, kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=True, bias_initializer='zeros'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_rate, seed=10))
    model.add(tf.keras.layers.Dense(units=16, activation=activation, kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=True, bias_initializer='zeros'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_rate, seed=10))
    model.add(tf.keras.layers.Dense(units=1, kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=True, bias_initializer='zeros'))

    # Save the model architecture to a JSON file
    model_json = model.to_json()
    architecture_filename = f"final_models/architectures/{str(nmg_conv)}_{activation}_architecture.json"
    with open(architecture_filename, 'w') as json_file:
        json_file.write(model_json)

    # Compiling the model and setting up training parameters
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps=X_train.shape[0]/batchsize, decay_rate=lr_decay, staircase=False)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(learning_rate=learningrate, beta_1=momentum), metrics=['accuracy', F1Score, Precision, Recall])

    # Print a summary of the model
    model.summary()

    # Define a ModelCheckpoint callback to save weights at the end of each epoch
    checkpoint_path = f"final_models/parameters/{str(nmg_conv)}_{activation}_model_weights_epoch_{epoch:02d}.h5"
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, period=1)

    # Record the start time
    start_time = time.time()
    
    # Define a callback to save weights
    # naming_conv_params = f"{nmg_conv}_{activation}"
    # save_weights_callback = SaveWeightsCallback(save_dir="final_models/parameters", naming_convention=naming_conv_params)

    # Training the model
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=batchsize, validation_data=(X_val, y_val), callbacks=[checkpoint_callback])

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    training_time = end_time - start_time

    # Format the time as hours, minutes, and seconds
    training_time_formatted = time.strftime("%H:%M:%S", time.gmtime(training_time))



    # Plotting/Saving Graphs
    # Extract the training history
    train_loss = history.history['loss']
    train_accuracy = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_accuracy']
    train_precision = history.history['Precision']
    train_recall = history.history['Recall']
    train_f1_score = history.history['F1Score']
    val_precision = history.history['val_Precision']
    val_recall = history.history['val_Recall']
    val_f1_score = history.history['val_F1Score']

    # Plot training and validation loss
    plt.figure(figsize=(16, 12))
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training and validation precision
    plt.subplot(2, 3, 4)
    plt.plot(range(1, len(train_precision) + 1), train_precision, label='Training Precision')
    plt.plot(range(1, len(val_precision) + 1), val_precision, label='Validation Precision')
    plt.title('Training and Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    # Plot training and validation recall
    plt.subplot(2, 3, 5)
    plt.plot(range(1, len(train_recall) + 1), train_recall, label='Training Recall')
    plt.plot(range(1, len(val_recall) + 1), val_recall, label='Validation Recall')
    plt.title('Training and Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    # Plot training and validation f1score
    plt.subplot(2, 3, 6)
    plt.plot(range(1, len(train_f1_score) + 1), train_f1_score, label='Training F1 Score')
    plt.plot(range(1, len(val_f1_score) + 1), val_f1_score, label='Validation F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    legend_text = f'Learning Rate: {learningrate}\nMomentum: {momentum}\nBatch Size: {batchsize}\nTotal Epochs: {epoch}\nActivation Function: {activation}\nLearning Rate Decay: {lr_decay}\nDrop Out Rate: {dropout_rate}\nTraining Time: {training_time_formatted}'
    plt.figtext(0.01, 0.01, legend_text, fontsize=10, va="bottom", ha="left")

    plt.tight_layout()

    # Save the figures to the specified directory
    figure_name = f"{nmg_conv}_{activation}_lowest_val_loss_{min(history.history['val_loss']):.4f}_FinalModel.png"  # Replace with your desired file name
    save_dir = "final_models/graphs"
    figure_path = os.path.join(save_dir, figure_name)
    plt.savefig(figure_path)
    plt.close()  # Close the figure to release resources

    # plt.show()