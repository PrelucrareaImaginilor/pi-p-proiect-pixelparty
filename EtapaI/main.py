import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR


def main():
    # Directory containing the fMRI data
    data_directory = 'TestData/train_tsv/train_tsv'
    files = os.listdir(data_directory)
    subjects = len(files)

    # Directory containing the additional data
    metadata_directory = 'TestData/metadata/training_metadata.csv'


    # Initialize an empty array for each subject's functional matrix
    n = 200 # the size of the square matrix
    FC = np.zeros((subjects, n*(n+1)//2)) # FC[i] = the upper triangular values from the subject's matrix
    ID = np.empty(subjects, dtype='U12') # ID[i] = the subject's ID 


    # Loop through each .tsv file to populate FC and ID
    for i, file in enumerate(files):
        FCMatrix = pd.read_csv(os.path.join(data_directory, file), sep='\t', header=None).values
        FC[i] = FCMatrix[np.triu_indices_from(FCMatrix)]
        ID[i] = file.split('_')[0][4:]


    # Make a dictionary with the additional info for each participant
    val = pd.read_csv(metadata_directory)
    data = list(zip(val['sex'].values, val['age'].values)) # Select only the 'sex' and 'age' columns
    data_id = val['participant_id'].values

    additional = {}
    for id, (sex, age) in zip(data_id, data):
        additional[id] = (sex, age)

    # Select the age of each subject in order
    y = np.zeros(subjects, dtype='double')
    for i in range(subjects):
        y[i] = additional[ID[i]][1] 

    # Monte Carlo partitioning
    partition = (len(FC)+1)//2 # Split in half
    
    xTrain = FC[0:partition]
    yTrain = y[0:partition]

    xTest = FC[partition:]
    yTest = y[partition:]

    # LINEAR REGRESSION
    startLinear = time.time()
    linear = LinearRegression()
    linear.fit(xTrain, yTrain)
    yPredict = linear.predict(xTest)
    endLinear = time.time()
    
    LinearRMSE = np.mean((yTest - yPredict)**2)**(1/2)
    print(f"RMSE linear: {LinearRMSE}")
    print (f"Time linear: {endLinear - startLinear}")

    # RIDGE REGRESSION
    startRidge = time.time()
    ridge = Ridge()
    ridge.fit(xTrain, yTrain)
    yPredictSVRRidge = ridge.predict(xTest)
    endRidge = time.time()

    RidgeRMSE = np.mean((yTest - yPredictSVRRidge) ** 2) ** (1 / 2)
    print(f"\nRMSE Ridge: {RidgeRMSE}")
    print(f"Time Ridge: {endRidge - startRidge}")

    # SVR LINEAR
    startSVRLin = time.time()
    svr = SVR(kernel = "linear")
    svr.fit(xTrain, yTrain)
    yPredictSVRLin = svr.predict(xTest)
    endSVRLin = time.time()

    SVRRMSELin = np.mean((yTest - yPredictSVRLin)**2)**(1/2)
    print(f"\nRMSE SVR linear: {SVRRMSELin}")
    print(f"Time SVR linear: {endSVRLin - startSVRLin}")

    # SVR POLYNOMIAL
    startSVRPoly = time.time()
    svr2 = SVR(kernel="poly")
    svr2.fit(xTrain, yTrain)
    yPredictSVRPoly = svr2.predict(xTest)
    endSVRPoly = time.time()

    SVRRMSEPoly = np.mean((yTest - yPredictSVRPoly) ** 2) ** (1 / 2)
    print(f"\nRMSE SVR poly: {SVRRMSEPoly}")
    print(f"Time SVR poly: {endSVRPoly - startSVRPoly}")

    # SVR GAUSSIAN
    startSVRGauss = time.time()
    svr3 = SVR(kernel="rbf")
    svr3.fit(xTrain, yTrain)
    yPredictSVRGauss = svr3.predict(xTest)
    endSVRGauss = time.time()

    SVRRMSEGauss = np.mean((yTest - yPredictSVRGauss) ** 2) ** (1 / 2)
    print(f"\nRMSE SVR Gauss: {SVRRMSEGauss}")
    print(f"Time SVR Gauss: {endSVRGauss - startSVRGauss}")

    # SVR SIGMOID
    startSVRSigm = time.time()
    svr4 = SVR(kernel="sigmoid")
    svr4.fit(xTrain, yTrain)
    yPredictSVRSigm = svr4.predict(xTest)
    endSVRSigm = time.time()

    SVRRMSESigm = np.mean((yTest - yPredictSVRSigm) ** 2) ** (1 / 2)
    print(f"\nRMSE SVR Sigmoid: {SVRRMSESigm}")
    print(f"Time SVR Sigmoid: {endSVRSigm - startSVRSigm}")


    # PLOTTING
    models = ["Linear", "Ridge", "SVR Linear", "SVR Poly", "SVR Gauss", "SVR Sigmoid"]
    RMSE_values = [LinearRMSE, RidgeRMSE, SVRRMSELin, SVRRMSEPoly, SVRRMSEGauss, SVRRMSESigm]

    # Time values for each model
    time_values = [
        endLinear - startLinear,
        endRidge - startRidge,
        endSVRLin - startSVRLin,
        endSVRPoly - startSVRPoly,
        endSVRGauss - startSVRGauss,
        endSVRSigm - startSVRSigm
    ]

    # Plot RMSE values
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(models, RMSE_values, color='skyblue')
    plt.xlabel("Regression Models")
    plt.ylabel("RMSE")
    plt.title("RMSE of Different Regression Models")
    plt.xticks(rotation=45)

    # Plot Time values
    plt.subplot(1, 2, 2)
    plt.bar(models, time_values, color='salmon')
    plt.xlabel("Regression Models")
    plt.ylabel("Time (s)")
    plt.title("Computation Time of Different Regression Models")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    #PLOT PREDICTIONS
    predictions = [yPredict, yPredictSVRRidge, yPredictSVRLin, yPredictSVRPoly, yPredictSVRGauss, yPredictSVRSigm]
    model_names = ["Linear", "Ridge", "SVR Linear", "SVR Poly", "SVR Gauss", "SVR Sigmoid"]
    
    # Loop over each model to plot its predicted vs. actual values
    for y_pred, name in zip(predictions, model_names):
        plt.figure(figsize=(8, 6))
        plt.scatter(yTest, y_pred, color='blue', label="Predicted vs Actual")
        plt.plot([yTest.min(), yTest.max()], [yTest.min(), yTest.max()], 'r--', lw=2, label="Perfect Fit")

        # Labels, title, and legend with increased font size
        plt.xlabel("Actual yTest", fontsize=14)
        plt.ylabel("Predicted yTest", fontsize=14)
        plt.title(f"{name} Model: Actual vs Predicted", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)

        plt.show()

if __name__ == '__main__':
    main()
