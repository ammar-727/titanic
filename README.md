### Idea
For this project, I created a simple deep-learning model from scratch and trained it by fientuning hyperparameters to maximize accuracy and minimize loss. The data set used was a simple tabular data of 1000 rows approx. and the model's architecture consisted of **Dense layers**, followed by **Batch Normalization layers** and **Dropout layers**. 

### Dataset
For this experiment, a simple tabular data was used from kaggle's competition 'Titanic Survival Prediction'. `'train.csv'` and `'test.csv'` consist of the tabular data that were used to create training/validation sets and test set, respectively.

### Scripts
`'main.py'` and `'predict.py'` were used to train and infer, respectively. Complete model architecture + tensorflow implementation of training the model can be seen in `'main.py'`. `'predict.py'` contains ensembling of 8 models that were finalized. The final answer is the average of all predictions that are made by the finalized models. 

### Tests
`'tests'` folder contain images of graphical representation of models training on training and validation sets. Testing starts with **20** models initialized, as can be seen in `'1st_batch_testing'` folder. As I proceed, I reject and approve models for next batch testing based on their results and hyperparameters.

The final batch (`'5th_batch_testing'`) contains a file `'Approved and Rejected Models List.txt'`, which shows the final 8 models that were approved, after finetuning hyperparameters. These 8 models will be used in ensemble to make final predictions.

**NOTE:** 1st, 2nd, and 3rd distribution of data means different distributions of data that were used to train the models.
`'Record.xlsx'` has the same information stored about hyperparameters in sequence of testing, as in `'tests'` folder, but in tabular form.

### Final Models (Ensemble) 
`'final_models/architectures'` contains architectures of all final models in **json** format.

`'final_models/parameters'` contains parameters of all final models in **h5** format.

`'final_models/graphs'` contains train and dev training graphs of all final models.

### Predictions
Predictions consists of csv files, that were used to submit in kaggle's competition. The test accuracy after training and inferring on test set was **77.75%**. 
