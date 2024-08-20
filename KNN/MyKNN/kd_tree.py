import numpy as np

class kd_tree:
    class _Node:
        def __init__(self, data):
            self.data = data
            self.left = None
            self.right = None


    def __init__(self, data):
        self.data = np.array(data)
        self.k = len(data[0])
        self.tree = self._build(self.data)


    def _build(self, data):
        if len(data) == 0:
            return None
        
        axis = np.argmax(np.var(data, axis=0))
        data = data[data[:, axis].argsort()]
        median = len(data) // 2

        node = self._Node(data[median])
        node.left = self._build(data[:median])
        node.right = self._build(data[median + 1:])
        return node
    

    def insert(self, point):
        self.data = np.vstack([self.data, point])
        self.tree = self._build(self.data)


    def remove(self, point):
        self.data = np.array([x for x in self.data if not np.array_equal(x, point)])
        self.tree = self._build(self.data)
        

    def knn(self, point, k):
        kpoints = []
        for i in range(k):
            kpoints.append(self._knn(self.tree, point, 0).data.tolist())
            self.remove(kpoints[-1])
        
        return kpoints
        
    
    def _knn(self, node, point, depth):
        if node is None:
            return None

        axis = depth % self.k

        if point[axis] < node.data[axis]:
            next_node = node.left
            opposite_node = node.right
        else:
            next_node = node.right
            opposite_node = node.left
        
        best = self._best(self._knn(next_node, point, depth + 1), node, point)

        if self._distance(point, node.data) < self._distance(point, best.data):
            best = self._best(self._knn(opposite_node, point, depth + 1), best, point)

        return best
        
    
    def _distance(self, a, b):
        sub = (a - b)
        return np.sqrt(np.dot(sub, sub.T))
    

    def _best(self, a, b, point):
        if a is None:
            return b
        if b is None:
            return a
        
        return a if self._distance(a.data, point) < self._distance(b.data, point) else b