#-------------------------------------------------------------------------------------------------------------
# AUTHOR: Dai (Paul) Vuong
# FILENAME: collaborative_filtering.py
# SPECIFICATION: Read the file trip_advisor_data.csv to make user-based recommendations. 
#                The goal is to predict the ratings of user 100 for the categories: galleries and restaurants. 
# FOR: CS 4200- Assignment #5
# TIME SPENT: 2 hours
#------------------------------------------------------------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter

df = pd.read_csv('trip_advisor_data.csv', sep=',', header=0) #reading the data by using the Pandas library ()
subset = df[["dance clubs", "juice bars", "museums", "resorts", "parks/picnic spots", "beaches", "theaters", "religious institutions"]]

#iterate over the other 99 users to calculate their similarity with the active user (user 100) according to their category ratings (user-item approach)
hashmap = {}
for i in range(0,99):
   # do this to calculate the similarity:
   #vec1 = np.array([[1,1,0,1,1]])
   #vec2 = np.array([[0,1,0,1,1]])
   #cosine_similarity(vec1, vec2)
   #do not forget to discard the first column (User ID) when calculating the similarities
   vec1 = np.array([subset.iloc[i]])
   vec2 = np.array([subset.iloc[99]])
   cos = cosine_similarity(vec1.reshape(1,-1), vec2.reshape(1,-1))
   hashmap[i] = cos[0][0]

#find the top 10 similar users to the active user according to the similarity calculated before
top10_similar = dict(sorted(hashmap.items(), key = itemgetter(1), reverse = True)[:10])

#Compute a prediction from a weighted combination of selected neighbors’ for both categories evaluated (galleries and restaurants)
r_galleries_numerator = 0
r_restaurants_numerator = 0
r_denominator = 0

for row, cos_value in top10_similar.items():
   r_galleries_numerator += cos_value * (float(df["galleries"][row]) - subset.iloc[row].mean())
   r_restaurants_numerator += cos_value * (float(df["restaurants"][row]) - subset.iloc[row].mean())
   r_denominator += cos_value

r_galleries = subset.iloc[99].mean() + (r_galleries_numerator / r_denominator)
r_restaurants = subset.iloc[99].mean() + (r_restaurants_numerator / r_denominator)

print("Weighted of galleries for User 100 is {}".format(r_galleries))
print("Weighted of restaurants for User 100 is {}".format(r_restaurants))