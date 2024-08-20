from knn import KNN
import numpy as np
import matplotlib.pyplot as plt

data = [[2,4], [8,-6], [6,2], [3,2], [-10,3]]

plt.scatter(np.array(data)[:, 0], np.array(data)[:, 1])
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.show()

knn = KNN(data, 2)

knn.insert([5,4])
knn.remove([3,2])

nearest = knn.knn([4,3])
print(nearest)