from collections import defaultdict
from math import log
class Node:
    def __init__(self, rows, cols):
        """
        @param:
        val: value for data_piece[split_col]
        split_col: column index for spliting the data, -1 means a leaf node
        rows: list, rows of data in this sub-tree
        cols: set, O(1) for removal ops (while O(N) for a list), excluding
            the last column, since its for labels
        label: prediction for labeling, only for leaves 

        @notice:
        a leaf node <=> node.split_col == -1
        """
        self.val = -1
        self.split_col = -1
        self.rows = rows
        self.cols = cols
        self.next = []
        self.label = None

class DecisionTreeClassifier:
    # For discrete features ONLY at present
    def __init__(self, data):
        self.data = data
        rows = [i for i in range(len(data))]
        cols = set([i for i in range(len(data[0])-1)])
        self.root = Node(rows, cols)

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
        node.split_col = split_col

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
            if len(v)<=0: continue
            kid_node = Node(v, cols_for_kids)
            kid_node.val = k
            node.next.append(kid_node)

    def majority_vote(self, node):
        """For a leaf node, vote for its labeling."""
        label_map = defaultdict(int)
        for r in node.rows:
            label_map[self.data[r][-1]] += 1
        #if len(node.rows)<=0:
        #    print node.val, node.split_col, node.label
        #    return
        most_common_label, cnt = max(label_map.items(), key=lambda (k,v): (v,k))
        return most_common_label, cnt

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
            if self.is_leaf(cur): 
                cur.label, _ = self.majority_vote(cur)
                continue
            self.split(cur)
            # I(X,Y) is under the threshold, thus leave as a leaf node
            if cur.split_col==-1: cur.label, _ = self.majority_vote(cur)
            for kid in cur.next:
                queue.append(kid)

    def bfs(self):
        print "val\tsplit_col\trows\tcols\tlabel\n"
        q = [self.root, ]
        while q:
            cur = q.pop(0)
            print cur.val, cur.split_col, cur.rows, cur.cols, cur.label
            for kid in cur.next:
                q.append(kid)

    def find_next_subtree(self, node, data, i):
        """Find which sub-tree this data (the i-th in dataset) belongs to."""
        if node.split_col==-1: return
        for v in node.next:
            if v.val==data[node.split_col]: return v
        # A new val for the split column found
        # This could occur while post-pruning,
        #   seeing new data from validation set
        new_kid = Node([i,], set([i for i in node.cols if i!=node.split_col]))
        new_kid.split_col = -1
        new_kid.label = data[-1]
        node.next.append(new_kid)
        return new_kid

    def predict(self, data):
        ans = []
        for i,d in enumerate(data):
            ans.append(self.predict_one(d,i))
        return ans

    def predict_one(self, data, i):
        """Given a piece of data (i-th of the dataset), predict its label."""
        cur = self.root
        while cur and cur.split_col != -1:
            cur = self.find_next_subtree(cur, data, i)
        return cur.label

    def dump_new_data(self, dataset):
        """Maintain the tree shape, dump new rows for each node.
        
        @params:
        dataset: 2D list with labels
        """
        self.data = dataset
        # remove rows for each node first
        queue = [self.root, ]
        while queue:
            cur = queue.pop(0)
            cur.rows = []
            for kid in cur.next:
                queue.append(kid)
        # dump new data based on the shape of the tree
        for row,data in enumerate(dataset):
            self.dump_one_data(data, row)

    def dump_one_data(self, data, i):
        """Dump one piece of data (i-th in the dataset) into the built tree."""
        cur = self.root
        while cur:
            cur.rows.append(i)
            cur = self.find_next_subtree(cur, data, i)
    
    def all_leaf_kids(self, node):
        for kid in node.next:
            if kid.split_col!=-1: return False
        return True

    def post_prune(self, validation_data):
        """Flow of post pruning.
        
        -> Dump validation data into the built tree
        -> Remove empty nodes
        -> Prune
        """
        self.dump_new_data(validation_data)
        self.remove_empty_nodes()
        self.prune(self.root)

    def remove_empty_nodes(self):
        """After dumping validation data into the built tree, remvoe empty 
        nodes, then do the actual pruning.
        """
        q = [self.root, ]
        while q:
            cur = q.pop(0)
            cur.next = [kid for kid in cur.next if len(kid.rows)>0]
            for kid in cur.next:
                q.append(kid)

    def prune(self, node):
        """Post prune main codes.
        
        -> Prune all its kids
        -> if it becomes a node whose kids are all leaves, try to cut its leaves 
        """
        for i,kid in enumerate(node.next):
            # If a non-leaf node gets no data while validation, remove this kid
            if kid and kid.split_col != -1:
                self.prune(kid)
        node.next = [kid for kid in node.next if kid!=None]
        if self.all_leaf_kids(node):
            self.cut(node)

    def cut(self, node):
        """Try to cut all the leaves of the node. If failed, maintain the shape.
        
        @notice:
        Suppose kids of node are all leaves."""
        err_cnt = 0 
        majority_label, max_samples_with_same_label = self.majority_vote(node)
        for leaf in node.next:
            for r in leaf.rows:
                err_cnt += (self.data[r][-1]!=leaf.label)
        err_after_cut = len(node.rows) - max_samples_with_same_label
        if err_after_cut > err_cnt: return
        # cut the leaves of the node
        node.split_col = -1
        node.next = []
        node.label = majority_label

    def score(self, data):
        """Get prediction precision score."""
        X, y = data, [j[-1] for j in data]
        pred = self.predict(X)
        precision = [pred[i]==y[i] for i in range(len(y))]
        return float(sum(precision)) / len(precision)
    
# unit test
def get_dataset(name="noisy10_train.ssv"):
    path = "D:\\Resource\\Courses\\cmu_ml\\dt_hw1\\dt_hw1\\"
    data = []
    with open(path+name) as fr:
        for line in fr.readlines():
            data.append([e for e in line.rstrip('\r\n').split(' ') if len(e)>0])
    for x in data:
        x[0], x[-1] = x[-1], x[0]
    return data[3:]

train = get_dataset()
valid = get_dataset("noisy10_valid.ssv")
test = get_dataset("noisy10_test.ssv")
# Working Flow
cc = DecisionTreeClassifier(train)
cc.fit()
print "before pruning: ", cc.score(test)
cc.post_prune(valid)
print "after pruning: ", cc.score(test)

# Results:
# before pruning: 0.818181818182
# after pruning: 0.896103896104
