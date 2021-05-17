#-------------------------------------------------------------------------
# AUTHOR: Dai (Paul) Vuong
# FILENAME: find_s.py
# SPECIFICATION: Find S Algorithm in ML
# FOR: CS 4200- Assignment #1
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
import csv

num_attributes = 4
db = []
print("\n The Given Training Data Set \n")

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

print("\n The initial value of hypothesis: ")
hypothesis = ['0'] * num_attributes #representing the most specific possible hypothesis
print(hypothesis)

#find the first positive training data in db and assign it to the vector hypothesis
first_positive_pos = 0
for i in range(len(db)):
  if db[i][-1] == 'Yes':
    hypothesis = db[i][:len(row)-1]
    first_positive_pos = i
    break
print("\n The first positive training data in db and assign it to the vector hypothesis: ")
print(hypothesis)

#find the maximally specific hypothesis according to your training data in db and assign it to the vector hypothesis (special characters allowed: "0" and "?")
for i in range(first_positive_pos+1, len(db)):
  if db[i][-1] == 'Yes':
    for j in range(len(hypothesis)):
      if hypothesis[j] != db[i][j]:
        hypothesis[j] = '?'

print("\n The Maximally Specific Hypothesis for the given training examples found by Find-S algorithm:")
print(hypothesis)

