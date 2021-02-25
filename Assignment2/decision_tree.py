#---------------------------------------------------------------------------------
# AUTHOR: Paul (Dai) Vuong
# FILENAME: decision_tree.py
# SPECIFICATION: training 3 datasets and test the model to find the least accuracy
# FOR: CS 4200- Assignment #2
# TIME SPENT: 1 hr 20 mins
#--------------------------------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTraining.append (row)

    age = {'Young':1, 'Prepresbyopic':2, 'Presbyopic':3}
    spectacle = {'Myope':1, 'Hypermetrope':2}
    astigmatism = {'Yes':1, 'No':2}
    tear = {'Normal':1, 'Reduced':2}
    recommended_lenses = {'Yes':1, 'No':2}

    #transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    for row in dbTraining:
        row_in_nums = [age[row[0]], spectacle[row[1]], astigmatism[row[2]], tear[row[3]]]
        X.append(row_in_nums)


    #transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    for row in dbTraining:
        Y.append(recommended_lenses[row[4]])


    #loop your training and test tasks 10 times here
    lowest_accuracy = 1
    for i in range (10):
        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)
        
        #read the test data and add this data to dbTest
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for j, row in enumerate(reader):
                if j > 0: #skipping the header
                    dbTest.append (row)

        TP, TN, FP, FN = 0, 0, 0, 0
        for data in dbTest:
            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            class_predicted = clf.predict([[age[data[0]], spectacle[data[1]], astigmatism[data[2]], tear[data[3]]]])[0]
            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            if data[4] == 'Yes' and class_predicted == 1:
                TP += 1
            if data[4] == 'No' and class_predicted == 1:
                FP += 1
            if data[4] == 'Yes' and class_predicted == 2:
                FN += 1
            if data[4] == 'No' and class_predicted == 2:
                TN += 1
        accuracy = (TP + TN) / (TP + TN + FP + FN) # Accuracy each time in loop of 10
        #find the lowest accuracy of this model during the 10 runs (training and test set)
        if accuracy < lowest_accuracy:
            lowest_accuracy = accuracy

    #print the lowest accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that:
            #final accuracy when training on contact_lens_training_1.csv: 0.2
            #final accuracy when training on contact_lens_training_2.csv: 0.3
            #final accuracy when training on contact_lens_training_3.csv: 0.4
    print('final accuracy when training on {}: {}'.format(ds, lowest_accuracy))




