from collections import defaultdict
from math import log
class Node:
    def __init__(self, val, rows, cols):
        """
        @param:
        val: column index for spliting the data, -1 means a leaf node
        rows: list, rows of data in this sub-tree
        cols: set, O(1) for removal ops (while O(N) for a list), excluding
            the last column, since its for labels
        """
        self.val = val
        self.rows = rows
        self.cols = cols
        self.next = []

class c:
    def __init__(self, data):
        self.data = data
        rows = [i for i in range(len(data))]
        cols = set([i for i in range(len(data[0])-1)])
        self.root = Node(-1, rows, cols)

    def entropy(self, rows):
        '''Calculate entropy for given rows of self.data'''
        labels, cnt, ent = defaultdict(int), len(rows), 0
        for r in rows:
            labels[self.data[r][-1]] += 1
        for _,val in labels.iteritems():
            prob = float(val) / cnt
            ent -= prob * log(prob, 2)
        return ent

    def split_entropy(self, node, col):
        '''Given a node, split its data by col. Return the entropy H(X|Y).'''
        split_map, HX_Y = defaultdict(list), 0.0
        for r in node.rows:
            split_map[self.data[r][col]].append(r)
        for k,v in split_map.iteritems():
            HX_Y += float(len(v)) / len(node.rows) * self.entropy(v)
        return HX_Y

    def split(self, node):
        '''Given a node, split it.
        
        -> Find the best col to split, asigned as node.val
        -> Construct sub-tree
        -> Split
        '''
        H_arr = [(col, self.split_entropy(node, col)) for col in node.cols]
        split_col, min_HX_Y = min(H_arr, key=lambda x: x[1])
        if min_HX_Y >= self.entropy(node.rows): return 
        node.val = split_col

        # O(1) for removal an item in a set, while O(N) for a list
        # You can't optimize copy section, but okay for searching
        node.cols.remove(split_col)
        cols_for_kids = set([col for col in node.cols])
        node.cols.add(split_col)

        split_map = defaultdict(list)
        # element would be a discrete type, thus apt for being a key in the map
        for r in node.rows:
            split_map[self.data[r][split_col]].append(r)
        for k,v in split_map.iteritems():
            kid_node = Node(-1, v, cols_for_kids)
            node.next.append(kid_node)

    def is_leaf(self, node):
        """Whether to stop here.
        
        See if all the data pieces belong to a single class or only 1 col exits.
        Note that if I(X,Y) is not good, a leaf node is ganranteed in split.
        """
        if len(node.cols)<=0: return True
        label_set = set([])
        for r in node.rows:
            label = self.data[r][-1]
            if label not in label_set:
                label_set.add(label)
            if len(label_set)>1: return False
        return True

    def fit(self):
        """Build up the tree."""
        queue = [self.root, ]
        while queue:
            cur = queue.pop(0)
            if self.is_leaf(cur): continue
            self.split(cur)
            for kid in cur.next:
                queue.append(kid)

    def bfs(self):
        q = [self.root, ]
        while q:
            cur = q.pop(0)
            print cur.val, cur.rows, cur.cols
            for kid in cur.next:
                q.append(kid)

# unit test
data = [[0,1,'no'],[0,1,'no'],[1,0,'no'],[1,1,'yes'],[1,1,'yes']]
cc = c(data)
cc.fit()
cc.bfs()
