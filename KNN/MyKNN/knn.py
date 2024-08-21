import numpy as np
import pandas as pd
from kd_tree import kd_tree

class KNN:
    def __init__(self, k, data=None, target=None):
        self.k = k
        if data is not None:
            self.fit(data, target)


    def fit(self, data, target):
        if isinstance(data, pd.DataFrame):
            self.data = data.to_numpy()
        elif isinstance(data, list):
            self.data = np.array(data)
        else:
            self.data = data

        if isinstance(target, pd.DataFrame):
            self.target = target.to_numpy()
        elif isinstance(target, list):
            self.target = np.array(target)
        else:
            self.target = target

        self.tree = kd_tree(self.data.tolist(), self.target.tolist())


    def insert(self, point, target):
        self.tree.insert(point, target)
    

    def remove(self, point):
        self.tree.remove(point)


    def knn(self, point):
        return self.tree.knn(point, self.k)
    

    def predict(self, points):
        if isinstance(points, pd.DataFrame):
            points = points.to_numpy()
        elif isinstance(points, list):
            points = np.array(points)
        else:
            points = points

        if len(points.shape) == 1:
            return self._predict(points)
        else:
            return [self._predict(point) for point in points]
    

    def _predict(self, point):
        nearest = self.knn(point)
        
        classes = {}
        for data in nearest:
            if data[1] in classes:
                classes[data[1]] += 1
            else:
                classes[data[1]] = 1

        return max(classes, key=classes.get)
    

    def print_tree(self):
        print(self.tree)