import numpy as np

class kd_tree:
    class _Node:
        def __init__(self, data, target):
            self.data = data
            self.target = target
            self.left = None
            self.right = None


    def __init__(self, data, target):
        self.k = len(data[0])
        self.tree = self._build(data, target)


    def _build(self, data, target, depth=0):
        if len(data) == 0:
            return None
        
        axis = depth % self.k
        
        data, target = zip(*sorted(zip(data, target), key=lambda x: x[0][axis]))
        median = len(data) // 2

        node = self._Node(data[median], target[median])
        node.left = self._build(data[:median], target[:median], depth + 1)
        node.right = self._build(data[median + 1:], target[median + 1:], depth + 1)

        return node
    

    def insert(self, point, target):
        self._insert(self.tree, point, target, 0)
    
    def _insert(self, node, point, target, depth):
        if node is None:
            return self._Node(point, target)
        
        axis = depth % self.k

        if point[axis] < node.data[axis]:
            node.left = self._insert(node.left, point, target, depth + 1)
        else:
            node.right = self._insert(node.right, point, target, depth + 1)
        
        return node
    

    def remove(self, point):
        self._remove(self.tree, point, 0)
    
    def _remove(self, node, point, depth):
        if node is None:
            return None
        
        axis = depth % self.k

        if node.data == point:
            if node.right is not None:
                min_node = self._min(node.right, axis, depth + 1)
                node.data = min_node.data
                node.target = min_node.target
                node.right = self._remove(node.right, min_node.data, depth + 1)
            elif node.left is not None:
                min_node = self._min(node.left, axis, depth + 1)
                node.data = min_node.data
                node.target = min_node.target
                node.right = self._remove(node.left, min_node.data, depth + 1)
            else:
                return None
        elif point[axis] < node.data[axis]:
            node.left = self._remove(node.left, point, depth + 1)
        else:
            node.right = self._remove(node.right, point, depth + 1)
        
        return node
    

    def _min(self, node, axis, depth):
        if node is None:
            return None
        
        if depth % self.k == axis:
            if node.left is None:
                return node
            return self._min(node.left, axis, depth + 1)
        
        left = self._min(node.left, axis, depth + 1)
        right = self._min(node.right, axis, depth + 1)

        if left is None and right is None:
            return node
        if left is None:
            return right
        if right is None:
            return left
        
        return left if left.data[axis] < right.data[axis] else right
        

    def knn(self, point, k):
        kpoints = []
        for i in range(k):
            best = self._knn(self.tree, point, 0)
            kpoints.append([best.data, best.target])
            self.remove(kpoints[-1][0])
        
        for point in kpoints:
            self.insert(point[0], point[1])

        return kpoints
        
    def _knn(self, node, point, depth):
        if node is None:
            return None
        
        axis = depth % self.k
        next_branch = None
        opposite_branch = None

        if point[axis] < node.data[axis]:
            next_branch = node.left
            opposite_branch = node.right
        else:
            next_branch = node.right
            opposite_branch = node.left
        
        best = self._best(self._knn(next_branch, point, depth + 1), node, point)
        if self._distance(point, best.data) > abs(point[axis] - node.data[axis]):
            best = self._best(self._knn(opposite_branch, point, depth + 1), best, point)
        
        return best
        
    
    def _distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    

    def _best(self, a, b, point):
        if a is None:
            return b
        if b is None:
            return a
        
        return a if self._distance(a.data, point) < self._distance(b.data, point) else b
    

    def __str__(self):
        return self._print(self.tree)
    
    def _print(self, node, depth=0):
        if node is None:
            return ''
        
        axis = depth % self.k
        left = self._print(node.left, depth + 1)
        right = self._print(node.right, depth + 1)

        return f'{node.data} {node.target}\n{left}{right}'