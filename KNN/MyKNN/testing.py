from knn import KNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = [[2,4], [8,-6], [6,2], [3,2], [-10,3]]

knn = KNN(3, data, [0, 5, 0, 5, 5])
print(knn.tree, "\n")

print(knn.predict([[4,3], [0,0], [10,10]]))