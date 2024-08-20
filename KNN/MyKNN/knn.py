import numpy as np
from kd_tree import kd_tree

class KNN:
    def __init__(self, data, k):
        self.data = np.array(data)
        self.k = k
        self.tree = kd_tree(data)


    def insert(self, point):
        self.data = np.vstack([self.data, point])
        self.tree.insert(point)

    
    def remove(self, point):
        self.data = np.array([x for x in self.data if not np.array_equal(x, point)])
        self.tree.remove(point)
    

    def knn(self, point):
        return self.tree.knn(point, self.k)