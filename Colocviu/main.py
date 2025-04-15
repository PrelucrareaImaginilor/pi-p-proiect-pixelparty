import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.ma.core import concatenate
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR

def SaveData(traindata_directory, testdata_directory, n):
    train_files = os.listdir(traindata_directory)
    test_files = os.listdir(testdata_directory)
    train_subjects = len(train_files)
    test_subjects = len(test_files)

    FCTrain = np.zeros((train_subjects, n * (n - 1) // 2))  # FC[i] = the upper triangular values from the subject's matrix
    FCTest = np.zeros((test_subjects, n * (n - 1) // 2))  # FC[i] = the upper triangular values from the subject's matrix
    IDTrain = np.empty(train_subjects, dtype='U12')  # ID[i] = ID of person i
    IDTest = np.empty(test_subjects, dtype='U12')  # ID[i] = ID of person i

    # Loop through each .tsv file to populate FC and ID
    for i, file in enumerate(train_files):
        FC = pd.read_csv(os.path.join(traindata_directory, file), sep='\t', header=None).values
        FCTrain[i] = FC[np.triu_indices_from(FC, 1)]
        IDTrain[i] = file.split('_')[0][4:]

    for i, file in enumerate(test_files):
        FC = pd.read_csv(os.path.join(testdata_directory, file), sep='\t', header=None).values
        FCTest[i] = FC[np.triu_indices_from(FC, 1)]
        IDTest[i] = file.split('_')[0][4:]

    # Save in files
    np.savetxt('FCTrain.txt', FCTrain, delimiter='\t')
    np.savetxt('FCTest.txt', FCTest, delimiter='\t')
    with open('IDTrain.txt', 'w') as f:
        for val in IDTrain:
            f.write(f'{val}\n')
    f.close()
    with open('IDTest.txt', 'w') as f:
        for val in IDTest:
            f.write(f'{val}\n')
    f.close()


def SaveMetadata(trainmetadata_directory, testmetadata_directory):
    # Make a dictionary with the metadata info for each participant
    train_val = pd.read_csv(trainmetadata_directory)
    test_val = pd.read_csv(testmetadata_directory)

    # Replace NAN values
    # Calculate the mean bmi for replacement
    train_bmi_clean = train_val['bmi'].values[~train_val['bmi'].isna()]
    test_bmi_clean = test_val['bmi'].values[~test_val['bmi'].isna()]
    concatenate_bmi = concatenate([train_bmi_clean, test_bmi_clean])
    mean_bmi = np.mean(concatenate_bmi)

    # Replace NAN's with unknown or the mean value
    train_val['sex'] =                train_val['sex'].fillna('Unknown')
    train_val['handedness'] =         train_val['handedness'].fillna('Unknown')
    train_val['bmi'] =                train_val['bmi'].fillna(22) # Mean of 18.5 and 22 (the limits of healthy bmi)
    train_val['race'] =               train_val['race'].fillna('Unknown')
    train_val['ethnicity'] =          train_val['ethnicity'].fillna('Unknown')
    train_val['parent_1_education'] = train_val['parent_1_education'].fillna('Unknown')
    train_val['parent_2_education'] = train_val['parent_2_education'].fillna('Unknown')

    test_val['sex'] = test_val['sex'].fillna('Unknown')
    test_val['handedness'] = test_val['handedness'].fillna('Unknown')
    test_val['bmi'] = test_val['bmi'].fillna(22)
    test_val['race'] = test_val['race'].fillna('Unknown')
    test_val['ethnicity'] = test_val['ethnicity'].fillna('Unknown')
    test_val['parent_1_education'] = test_val['parent_1_education'].fillna('Unknown')
    test_val['parent_2_education'] = test_val['parent_2_education'].fillna('Unknown')

    # Map the numerical values between -1 and 1
    train_bmi_values = train_val['bmi'].values.astype(float)
    test_bmi_values = test_val['bmi'].values.astype(float)
    min_bmi = min(concatenate_bmi)
    max_bmi = max(concatenate_bmi)
    train_val['bmi'] = 2 * (train_bmi_values - min_bmi) / (max_bmi - min_bmi) - 1
    test_val['bmi'] = 2 * (test_bmi_values - min_bmi) / (max_bmi - min_bmi) - 1

    train_data = list(zip(train_val['sex'].values,
                    train_val['handedness'].values,
                    train_val['bmi'].values,
                    train_val['race'].values,
                    train_val['ethnicity'].values,
                    train_val['parent_1_education'].values,
                    train_val['parent_2_education'].values,
                    train_val['age'].values))

    test_data = list(zip(test_val['sex'].values,
                          test_val['handedness'].values,
                          test_val['bmi'].values,
                          test_val['race'].values,
                          test_val['ethnicity'].values,
                          test_val['parent_1_education'].values,
                          test_val['parent_2_education'].values))

    train_data_id = train_val['participant_id'].values
    test_data_id = test_val['participant_id'].values

    # Encode the categorical data
    categorical_columns = [0, 1, 3, 4, 5, 6]
    train_numerical_columns = [2, 7]
    test_numerical_columns = [2]

    train_data_array = np.array(train_data)
    test_data_array = np.array(test_data)

    train_categorical_data = train_data_array[:, categorical_columns]
    test_categorical_data = test_data_array[:, categorical_columns]

    train_numerical_data = train_data_array[:, train_numerical_columns].astype(float)
    test_numerical_data = test_data_array[:, test_numerical_columns].astype(float)

    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')  # drop='first' to avoid multicollinearity
    train_encoded_data = encoder.fit_transform(train_categorical_data)
    test_encoded_data = encoder.transform(test_categorical_data)
    train_encoded_data[train_encoded_data == 0] = -1
    test_encoded_data[test_encoded_data == 0] = -1

    train_final_data = np.hstack([train_encoded_data, train_numerical_data])
    test_final_data = np.hstack([test_encoded_data, test_numerical_data])

    # Save the data
    with open('trainmetadata.txt', 'w') as f:
        for id, data in zip(train_data_id, train_final_data):
            data_str = ' '.join(map(str, data))
            f.write(f'{id} {data_str}\n')
    f.close()

    with open('testmetadata.txt', 'w') as f:
        for id, data in zip(test_data_id, test_final_data):
            data_str = ' '.join(map(str, data))
            f.write(f'{id} {data_str}\n')
    f.close()


'''
def SexAgePartition (FC, metadata, ID, num_subjects, partition):
    sex = np.zeros(num_subjects, dtype='int')
    age = np.zeros(num_subjects, dtype='double')
    for i in range(num_subjects):
        if (metadata[ID[i]][0] == 'Male'):
            sex[i] = 1
        else:
            sex[i] = 0
        age[i] = metadata[ID[i]][1]

    age_bins = np.digitize(age, bins = [5, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    stratify_col = np.char.add(sex.astype(str), "_" + age_bins.astype(str))
    train_index, test_index = train_test_split(np.arange(num_subjects), train_size=partition, stratify=stratify_col, random_state=42)
    xTrain = FC[train_index]
    xTest = FC[test_index]
    yTrain = age[train_index]
    yTest = age[test_index]

    return xTrain, yTrain, xTest, yTest
'''


def CreateSubmission(ID, yPredict, model_name):
    submission = pd.DataFrame({
        "participant_id": ID,
        "age": yPredict
    })
    submission.to_csv(f'submission_{model_name}.csv', index=False)


def TrainModel(x, y, model, k):
    rmse_scores = []
    kf = KFold(n_splits=k, random_state=42, shuffle=True)

    i = 1
    for train_index, test_index in kf.split(x, y):
        xTrain, xTest = x[train_index], x[test_index]
        yTrain, yTest = y[train_index], y[test_index]

        # Apply the model
        model.fit(xTrain, yTrain)
        yPredict = model.predict(xTest)

        RMSEfold = np.mean((yTest - yPredict) ** 2) ** (1 / 2)
        rmse_scores.append(RMSEfold)

        print(f'\nRMSE fold {i}: {RMSEfold}')
        i = i+1

    meanRMSE = np.mean(rmse_scores)
    print(f'\nRMSE mean: {meanRMSE}')

    # Plot RMSE across folds
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, k + 1), rmse_scores, marker='o', linestyle='-', color='b')
    plt.axhline(y=meanRMSE, color='r', linestyle='--', label=f"Mean RMSE: {meanRMSE:.2f}")
    plt.title("RMSE Across Folds")
    plt.xlabel("Fold Number")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid()
    plt.show()


def SubmitModel(xTrain, yTrain, xTest, IDTest, model, name):
    model.fit(xTrain, yTrain)
    yPredict = model.predict(xTest)
    CreateSubmission(IDTest, yPredict, name)


def main():
    # Directory containing the fMRI data
    traindata_directory = 'TestData/train_tsv/train_tsv'
    testdata_directory = 'TestData/Test_tsv/Test_tsv'

    train_files = os.listdir(traindata_directory)
    test_files = os.listdir(testdata_directory)
    train_subjects = len(train_files)
    test_subjects = len(test_files)

    # Directory containing the additional data
    trainmetadata_directory = 'TestData/metadata/training_metadata.csv'
    testmetadata_directory = 'TestData/metadata/test_metadata.csv'

    # Initialize an empty array for each subject's functional matrix
    n = 200 # 200x200 matrix

    # Save the data
    #SaveData(traindata_directory, testdata_directory, n)
    #SaveMetadata(trainmetadata_directory, testmetadata_directory)

    # Load from the files
    FCTrain = np.loadtxt('FCTrain.txt', delimiter='\t')
    with open('IDTrain.txt', 'r') as f:
        IDTrain = np.array([line.strip() for line in f])
    f.close()

    FCTest = np.loadtxt('FCTest.txt', delimiter='\t')
    with open('IDTest.txt', 'r') as f:
        IDTest = np.array([line.strip() for line in f])
    f.close()


    trainmetadata = {}
    testmetadata = {}
    with open('trainmetadata.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            id = parts[0]
            data = np.array(parts[1:-1]).astype(float)
            age = float(parts[-1])
            trainmetadata[id] = (data, age)
    f.close()

    with open('testmetadata.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            id = parts[0]
            data = np.array(parts[1:]).astype(float)
            testmetadata[id] = data
    f.close()


    # ------------------------------------------------------------------------------
    # CLUSTERIZATION
    kmeans = KMeans(n_clusters=2, random_state=0, algorithm='elkan').fit(FCTrain)
    labels = (kmeans.labels_).reshape(-1, 1)
    pred = kmeans.predict(FCTest).reshape(-1, 1)
    # ------------------------------------------------------------------------------


    # Concatenate metadata to the value vectors
    y = np.zeros(train_subjects, dtype='double')
    Train = np.zeros((train_subjects, n * (n - 1) // 2 + 1 + len(testmetadata[IDTest[0]])))
    Test = np.zeros((test_subjects, n * (n - 1) // 2 + 1 + len(testmetadata[IDTest[0]])))
    for i in range(train_subjects):
        Train[i] = np.concatenate([labels[i], trainmetadata[IDTrain[i]][0], FCTrain[i]])
        y[i] = trainmetadata[IDTrain[i]][1]

    for i in range(test_subjects):
        Test[i] = np.concatenate([pred[i], testmetadata[IDTest[i]], FCTest[i]])


    # Monte Carlo partitioning
    #xTrain, yTrain, xTest, yTest = SexAgePartition(FCTrain, trainmetadata, IDTrain, train_subjects, 0.8)
    #xTrain, xTest, yTrain, yTest = train_test_split(Train, y, train_size=0.8, random_state=42)

    # SVR BEST PARAMS
    paramSVR = {
        'C': 100,
        'epsilon': 0.1,
        'gamma': 'auto',
        'kernel': 'rbf'
    }
    svr = SVR(**paramSVR)

    TrainModel(Train, y, svr, 5)
    #SubmitModel(Train, y, Test, IDTest, svr, 'rbfSVRparent_education')

    # LINEAR REGRESSION
    '''
    linear = LinearRegression()
    TrainModel(Train, y, linear, 5)
    '''

    # RIDGE REGRESSION
    '''
    ridge = Ridge()
    TrainModel(Train, y, ridge, 5)
    '''

    # SVR LINEAR
    '''
    paramSVR = {
        'C': 0.1,
        'epsilon': 0.01,
        'kernel': 'linear'
    }
    svr = SVR(**paramSVR)
    TrainModel(Train, y, svr, 5)
    '''

    # SVR POLYNOMIAL
    '''
    svr2 = SVR(kernel="poly")
    TrainModel(Train, y, svr2, 5)
    '''

    # SVR GAUSSIAN
    '''
    svr3 = SVR(kernel="rbf")
    TrainModel(Train, y, svr3, 5)
    '''

    # SVR SIGMOID
    '''
    svr4 = SVR(kernel="sigmoid")
    TrainModel(Train, y, svr4, 5)
    '''


if __name__ == '__main__':
    main()
