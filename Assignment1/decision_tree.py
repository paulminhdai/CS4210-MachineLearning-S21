#-------------------------------------------------------------------------
# AUTHOR: Dai (Paul) Vuong
# FILENAME: decision_tree.py
# SPECIFICATION: Calculate entropy/gain to set up a decision tree
# FOR: CS 4200- Assignment #1
# TIME SPENT: 45 minutes
#------------------------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays
#importing some Python libraries

from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transfor the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
for row in db:
  row_in_nums = []
  if row[0] == "Young":
    row_in_nums.append(1)
  elif row[0] == "Prepresbyopic":
    row_in_nums.append(2)
  elif row[0] == "Presbyopic":
    row_in_nums.append(3)

  if row[1] == "Myope":
    row_in_nums.append(1)
  elif row[1] == "Hypermetrope":
    row_in_nums.append(2)

  if row[2] == "Yes":
    row_in_nums.append(1)
  elif row[2] == "No":
    row_in_nums.append(2)
    
  if row[3] == "Normal":
    row_in_nums.append(1)
  elif row[3] == "Reduced":
    row_in_nums.append(2)

  X.append(row_in_nums)


#transfor the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
for i in range(len(db)):
  if db[i][-1] == "Yes":
    Y.append(1)
  else:
    Y.append(2)


# # #fiiting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

# #plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()


