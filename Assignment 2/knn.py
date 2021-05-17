#-------------------------------------------------------------------------
# AUTHOR: Paul (Dai) Vuong
# FILENAME: The LOO-CV error rate for 1NN
# SPECIFICATION: Test and calculate the error rate by sequential taking 
#                one instance in dataset as a test sample.
# FOR: CS 4200- Assignment #2
# TIME SPENT: 30 mins
#------------------------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

classification = {'+':1, '-':2}
error_count = 0 # count errors

#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]
    X = list()
    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]
    Y = list()
    for j, instance1 in enumerate(db):
        if i != j:
            X.append([int(instance1[0]), int(instance1[1])])
            Y.append(classification[instance1[-1]])

    #store the test sample of this iteration in the vector testSample
    testSample = [int(instance[0]), int(instance[1])]
    
    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    class_predicted = clf.predict([testSample])[0]


    #compare the prediction with the true label of the test instance to start calculating the error rate.
    if (class_predicted == 1 and classification[instance[2]] == 2) or \
        (class_predicted == 2 and classification[instance[2]] == 1):
        error_count += 1


#print the error rate
error_rate = error_count / len(db)
print('Error rate is: {}'.format(error_rate))






