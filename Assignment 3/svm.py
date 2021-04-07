#-------------------------------------------------------------------------
# AUTHOR: Dai (Paul) Vuong
# FILENAME: svm.py
# SPECIFICATION: combination of four SVM hyperparameters (c, degree, kernel, 
# and decision_function_shape) leads to the best prediction performance
# FOR: CS 4200- Assignment #3
# TIME SPENT: 1.5 hours
#------------------------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import svm
import csv

dbTraining = []
dbTest = []
X_training = []
Y_training = []
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0
best_params = []

#reading the data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
  reader = csv.reader(trainingFile)
  for i, row in enumerate(reader):
      X_training.append(row[:-1])
      Y_training.append(row[-1])

#reading the data in a csv file
with open('optdigits.tes', 'r') as testingFile:
  reader = csv.reader(testingFile)
  for i, row in enumerate(reader):
      dbTest.append (row)

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape

for c_value in c : #iterates over c
    for degree_value in degree: #iterates over degree
        for kernel_value in kernel: #iterates kernel
           for shape in decision_function_shape: #iterates over decision_function_shape

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape as hyperparameters. For instance svm.SVC(c=1)
                clf = svm.SVC(C=c_value, degree=degree_value, kernel=kernel_value, decision_function_shape=shape)

                #Fit Random Forest to the training data
                clf.fit(X_training, Y_training)

                #make the classifier prediction for each test sample and start computing its accuracy
                true_count = 0
                for testSample in dbTest:
                    class_predicted = clf.predict([testSample[:-1]])[0]
                    if class_predicted == testSample[-1]:
                        true_count += 1
                accuracy = true_count / len(dbTest)
                #check if the calculated accuracy is higher than the previously one calculated. If so, update update the highest accuracy and print it together with the SVM hyperparameters
                #Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                if accuracy > highestAccuracy:
                    highestAccuracy = accuracy
                    best_params = [c_value, degree_value, kernel_value, shape]
                    print("Highest SVM accuracy so far: {:f}, Parameters: a={:d}, degree={:d}, kernel= {:s}, decision_function_shape = {:s}".format(highestAccuracy, c_value, degree_value, kernel_value, shape))

#print the final, highest accuracy found together with the SVM hyperparameters
#Example: "Highest SVM accuracy: 0.95, Parameters: a=10, degree=3, kernel= poly, decision_function_shape = 'ovr'"
print("=> Highest SVM accuracy: {:f}, Parameters: a={:d}, degree={:d}, kernel= {:s}, decision_function_shape = {:s}".format(highestAccuracy, best_params[0], best_params[1], best_params[2], best_params[3]))
