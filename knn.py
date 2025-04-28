#-------------------------------------------------------------------------
# AUTHOR: Sheldin Lau
# FILENAME: knn.py
# SPECIFICATION: finding the best hyperparameters for a KNN Classifier
# FOR: CS 5990- Assignment #4
# TIME SPENT: 2 Hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

#reading the training data
#reading the test data
#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')
df = pd.read_csv('weather_training.csv', sep=',', header=0)
data_training = np.array(df.values)[:,-1]

X_training = np.array(df.values)[:,1:-1].astype('f')
y_training = np.array(df.values)[:,-1].astype('f')

y_training = np.digitize(y_training, classes)

df = pd.read_csv('weather_test.csv', sep=',', header=0)
data_training = np.array(df.values)[:,-1]

X_test = np.array(df.values)[:,1:-1].astype('f')
y_test = np.array(df.values)[:,-1].astype('f')

y_test = np.digitize(y_test, classes)

#loop over the hyperparameter values (k, p, and w) ok KNN
#--> add your Python code here
highest_accuracy = 0
for n_neighbor in k_values:
    for pval in p_values:
        for weight in w_values:

            #fitting the knn to the data
            #--> add your Python code here

            #fitting the knn to the data
            # print(n_neighbor, pval, weight)
            clf = KNeighborsClassifier(n_neighbors=n_neighbor, p=pval, weights=weight)
            clf = clf.fit(X_training, y_training)

            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            #--> add your Python code here
            correctPredictions = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                predicted_value = clf.predict([x_testSample])
                real_value = y_testSample
                # print(predicted_value, real_value)
                difference = 100*abs(predicted_value - real_value)/abs(real_value)
                # print(difference)
                if difference <= 15:
                    correctPredictions += 1

            tmp = clf.predict(X_test)
            # print(tmp)


            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            #--> add your Python code here

            accuracy = correctPredictions/len(X_test)
            if accuracy > highest_accuracy:
                print("Highest KNN accuracy so far:" + str(accuracy) + ", Parameters: k=" + str(n_neighbor) + ", p=" +  str(pval) + ", w= "+  weight)
                highest_accuracy = accuracy






