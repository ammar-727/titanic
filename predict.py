from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score



# Importing and cleaning data
# importing csv data
# Step 1: Extract training data from train.csv in pandas DataFrame
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# Step 2: Feature engineering training data
# Removing unnecessary rows for train data
titanic_useable_data_train = train_data[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
titanic_useable_data_train = titanic_useable_data_train.copy()

# Adding "missing_age" column where value=1 if "age" is given and value=0 if "age" is not given and replacing missing values in the 'Age' column with 0 for train data
titanic_useable_data_train['Age_missing'] = titanic_useable_data_train['Age'].isna().astype(int)
titanic_useable_data_train['Age'] = titanic_useable_data_train['Age'].fillna(0)

# Adding a new column "titles" that consists only title part form the whole name. e.g. "Braund, Mr. Owen Harris" becomes "Mr". Also removing the column "Name" for train data
titanic_useable_data_train['Titles'] = titanic_useable_data_train['Name'].str.split(', ').str[1].str.split('.').str[0]
titanic_useable_data_train = titanic_useable_data_train.drop('Name', axis=1)

# Using one-hot-encoding foe "Sex", "Embarked", and "Titles"
titanic_useable_data_train = pd.get_dummies(titanic_useable_data_train, columns=['Sex', 'Embarked', 'Titles'])

# Extract 'PassengerId' column to a separate list
passenger_ids_train = titanic_useable_data_train.pop('PassengerId').tolist()

# Normalizing Features
titanic_features_norm_train = (titanic_useable_data_train - titanic_useable_data_train.mean()) / titanic_useable_data_train.std()


# Step 3: Feature engineering test data
# Removing unnecessary rows for test data
titanic_useable_data_test = test_data[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
titanic_useable_data_test = titanic_useable_data_test.copy()

# Adding "missing_age" column where value=1 if "age" is given and value=0 if "age" is not given and replacing missing values in the 'Age' column with 0 for test data
titanic_useable_data_test['Age_missing'] = titanic_useable_data_test['Age'].isna().astype(int)
titanic_useable_data_test['Age'] = titanic_useable_data_test['Age'].fillna(0)

# Adding a new column "titles" that consists only title part form the whole name. e.g. "Braund, Mr. Owen Harris" becomes "Mr". Also removing the column "Name" for test data
titanic_useable_data_test['Titles'] = titanic_useable_data_test['Name'].str.split(', ').str[1].str.split('.').str[0]
titanic_useable_data_test = titanic_useable_data_test.drop('Name', axis=1)

# Using one-hot-encoding foe "Sex", "Embarked", and "Titles"
titanic_useable_data_test = pd.get_dummies(titanic_useable_data_test, columns=['Sex', 'Embarked', 'Titles'])

# Extract 'PassengerId' column to a separate list
passenger_ids_test = titanic_useable_data_test.pop('PassengerId').tolist()

# Normalizing Features
titanic_features_norm_test = (titanic_useable_data_test - titanic_useable_data_test.mean()) / titanic_useable_data_test.std()


# Step 4: Compare headers of all columns of training and test data frames
train_columns = set(titanic_features_norm_train.columns)
test_columns = set(titanic_features_norm_test.columns)


#Step 5: Fill missing columns and remmove extra columns in the test data and re-order columns, according to the training data
# Get the common columns between training and test data frames
common_columns = list(train_columns.intersection(test_columns))

# Keep only the common columns in the test data frame
titanic_features_norm_test = titanic_features_norm_test[common_columns].fillna(0)

# Reorder columns of the test data frame based on the order of the training data frame
titanic_features_norm_test = titanic_features_norm_test.reindex(columns=titanic_features_norm_train.columns, fill_value=0)


# Step 6: Converting Pandas DataFrame to numpy array
titanic_features_to_predict = titanic_features_norm_test.values



# Function to predict
def load_and_predict(model_name, epoch_to_load, new_data):
    # Load the model architecture
    with open(f"final_models/architectures/{model_name}_architecture.json", 'r') as json_file:
        model_json = json_file.read()

    # Load the model
    model = keras.models.model_from_json(model_json)

    # Load the weights of the model from the specified epoch
    model.load_weights(f"final_models/parameters/{model_name}_model_weights_epoch_{epoch_to_load:02d}.h5")

    # Make predictions on the new data
    predictions = model.predict(new_data)

    # Return the predictions
    return predictions



# Predicting on test data
models_name = ['2_tanh', '4_tanh', '6_relu', '6_tanh', '9_tanh', '11_relu', '13_relu']
epoch_to_load = 15000
predictions_org = []

for i in range (len(models_name)):
    model_name = f"{models_name[i]}"
    predictions = load_and_predict(model_name, epoch_to_load, titanic_features_to_predict)
    predictions_org.append(predictions)

# Converting normal array to numpy array
predictions_binary_array = np.array(predictions_org)

# Calculate the mean along axis 0 (columns)
average_predictions = np.mean(predictions_binary_array, axis=0)

# Convert the average predictions to binary based on the mean threshold
average_predictions_binary = np.where(average_predictions >= 0.5, 1, 0)
predictions_binary = [1 if pred >= 0.5 else 0 for pred in average_predictions_binary]



#Create a DataFrame with 'PassengerId' and predictions
result_df = pd.DataFrame({'PassengerId': passenger_ids_test, 'Survived': predictions_binary})

#Save the DataFrame to a CSV file
result_csv_path = f"predictions/final_predictions_on_test_data_2nd.csv"
result_df.to_csv(result_csv_path, index=False)

print(f"\nPredictions saved to {result_csv_path}")