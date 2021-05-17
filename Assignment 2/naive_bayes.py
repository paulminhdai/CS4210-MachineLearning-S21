#---------------------------------------------------------------------------------------
# AUTHOR: Paul (Dai) Vuong
# FILENAME: applied naive bayes algorithm to predict play tennis with weather conditions
# SPECIFICATION: train data with Naive Bayes algorithm and predict test data 
#                if if the classification confidence is >= 0.75
# FOR: CS 4200- Assignment #2
# TIME SPENT: 50 mins
#--------------------------------------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

dbTraining = []
dbTesting = []

Outlook = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
Temperature = {'Hot': 1, 'Mild': 2, 'Cool': 3}
Humidity = {'High': 1, 'Normal': 2}
Wind = {'Strong': 1, 'Weak': 2}
classification = {'Yes':1, 'No':2}

#reading the training data
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTraining.append (row)

#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
X = list()
#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
Y = list()
for row in dbTraining:
    X.append([Outlook[row[1]], Temperature[row[2]], Humidity[row[3]], Wind[row[4]]])
    Y.append(classification[row[-1]])


#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the data in a csv file
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTesting.append (row)

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions.
for row in dbTesting:
    test_case = [Outlook[row[1]], Temperature[row[2]], Humidity[row[3]], Wind[row[4]]]
    predicted = clf.predict_proba([test_case])[0]
    most_probable = 0
    most_probable_class = ""
    if predicted[0] > predicted[1]:
        most_probable = predicted[0]
        most_probable_class += "Yes"
    else:
        most_probable = predicted[1]
        most_probable_class += "No"

    if most_probable >= 0.75:
        print("{:15s}{:15s}{:15s}{:15s}{:15s}{:15s}{:10.2f}".format(row[0], row[1], row[2], row[3], row[4], most_probable_class, most_probable)) 


