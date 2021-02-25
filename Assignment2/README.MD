### Analyst decision tree, KNN algorithm, Naive Bayes strategy

1. Complete the python program (decision_tree.py) that will read the files contact_lens_training_1.csv, contact_lens_training_2.csv, and contact_lens_training_3.csv. Each of those training sets has a different number of instances. You will observe that now the trees are being created setting the parameter `max_depth = 3`, which it is used to define the maximum depth of the tree (pre-pruning strategy) in sklearn. Your goal is to train, test and output the performance of the models created by using each training set on the test set provided (contact_lens_test.csv). You must repeat this process 10 times (train and test by using a different training set), choosing the lowest accuracy as the classification performance of each model. 

2. Complete the python program (knn.py) that will read the file binary_points.csv and output the LOO-CV error rate for 1NN (same answer of part a). 

3. Complete the Python program (naïve_bayes.py) that will read the file weather_training.csv (training set) and output the classification of each test instance from the file weather_test (test set) if the classification confidence is >= 0.75. Sample of output: 

  |Day| Outlook| Temperature| Humidity| Wind| PlayTennis| Confidence|
  
  |---|---|---|---|---|---|
  
  |D15| Sunny| Hot| High| Weak| No| 0.86|
  
  |D16| Sunny| Mild| High| Weak| Yes| 0.78| 

