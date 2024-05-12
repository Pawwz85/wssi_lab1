import numpy as np
from collections import Counter, deque


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def is_leaf_node(self):
    return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, x, y):
        self.root = Node()
        self._grow_tree(self.root, x, y)

    def is_one_class_label(self, y):
        if len(y) == 0:
            return True
        val = y[0]
        for label in y:
            if label != val:
                return False
        return True

    def _grow_tree(self, node, x, y, depth=0):

        if (depth > self.max_depth and depth > self.min_samples_split) \
                or len(y) == 0 or self.is_one_class_label(y):
            if len(y) > 0:
                node.value = Counter(y).most_common(1)[0][0]
            else:
                node.value = 0
            return

        feat_idxs = np.random.choice(len(x[0]), self.n_features, replace=False) if self.n_features else np.arange(
            len(x[0]))

        best_feat_idx = self._best_split(x, y, feat_idxs)
        best_thresh = np.median(x[:, best_feat_idx])
        node.feature = best_feat_idx
        node.threshold = best_thresh
        left, right = self._split(x[:, best_feat_idx], best_thresh)

        left_x = np.array([x[i] for i in left])
        left_y = np.array([y[i] for i in left])
        right_x = np.array([x[i] for i in right])
        right_y = np.array([y[i] for i in right])
        node.left = Node()
        node.right = Node()
        self._grow_tree(node.left, left_x, left_y, depth + 1)
        self._grow_tree(node.right, right_x, right_y, depth + 1)

    def _best_split(self, x, y, feat_idxs):
        result = np.argmax(
            [self._information_gain(y, x[:, feat_id], np.median(x[:, feat_id])) for feat_id in feat_idxs])
        return feat_idxs[result]

    def _information_gain(self, y, X_column, threshold):
        cls_lesser = [y[i] for i, x in enumerate(X_column) if x < threshold]
        cls_more = [y[i] for i, x in enumerate(X_column) if x >= threshold]
        return self._entropy(y) - self._entropy(cls_lesser) - self._entropy(cls_more)

    def _split(self, X_column, split_thresh):
        less = [i for i, x in enumerate(X_column) if x < split_thresh]
        more = [i for i, x in enumerate(X_column) if x >= split_thresh]
        return less, more

    def _entropy(self, y):
        if len(y) == 0:
            return 1
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _traverse_tree(self, x, node: Node):
        if is_leaf_node(node):
            return node.value
        else:
            child = node.left if x[node.feature] < node.threshold else node.right
            return self._traverse_tree(x, child)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
