import numpy as np
import matplotlib.pyplot as plt
import json
import unionfindweighted as uf

class DendrogramTree(object):
    def __init__(self):
        self.root = None
    
    def construct_dendrogram(self, keys, data):
        """
        
        Parameters
        ----------
        keys : list
            all animal names
        data : dictionary(
                string ("animal1,animal2"): int (needleman wunsch score)
            )

        Returns
        -------
        calls draw methods

        """
        
        #   1. make a list of leaf nodes, one for each animals
        
        #   O(N)
        leaf_nodes = []
        for key in keys:
            leaf_nodes.append(TreeNode(key))
            
        
        #   2. sort scores in descending order
        #   This is O(N^2 logN) right now, need to reconsider w/ Prim's algorithm
        sorted_data = {r: data[r] for r in sorted(data, key = data.get, reverse = True)}
        
        
              
        #   3. check to see if each pair is in same connected component and union if able
        ufObject = uf.UnionFind(len(keys))
        
        # convert animals into unique numbers for unionfind
        #   O(N)
        animal2num = {s:i for i,s in enumerate(keys)}
        
        #dictionary for keeping track of parents in tree
        roots = {}
        
        # each leaf node starts as its own root
        #   O(N)
        for n in leaf_nodes:
            roots[animal2num[n.key]] = n
        
        """
        ---------------------------
              Data Structures
        ---------------------------
        
        leaf_nodes : list
            a list of nodes where the node key is an animal name
            
        sorted_data : dictionary (data, but sorted by value descending)
        
        animal2num : dictionary(
                string (single animal name): int (unique id)
            )
        
        roots : dictionary(
                int (animal id): node (highest order node in a leaf's component)
            )
                
        """
        
        #   O(N^2)
        for pair in sorted_data.keys():
            pair_to_check = pair.split(',')
            if not ufObject.find(animal2num[pair_to_check[0]], animal2num[pair_to_check[1]]):
                
                #everything in here is 0(1) time
                
                # new internal node with similarity score as key
                node = TreeNode(sorted_data[pair])
                
                # set children of new internal node
                node.left = roots[ufObject.root(animal2num[pair_to_check[0]])]
                node.right = roots[ufObject.root(animal2num[pair_to_check[1]])]
                
                # union the animals together
                ufObject.union(animal2num[pair_to_check[0]], animal2num[pair_to_check[1]])
                
                # reassign root of both animals' leaf nodes to be the new internal node
                # set dictionary to root of both leaf nodes
                roots[ufObject.root(animal2num[pair_to_check[0]])] = node
                roots[ufObject.root(animal2num[pair_to_check[1]])] = node
                
                
        #the last node we created is the root of the tree
        
        self.root = node
        
        
        #get the highest similarity score to serve as x coordinate for leaf nodes
        maxScore = next(iter(sorted_data.values()))
        
        
        #draw the tree
        self.root.draw_setup(self.root.key, maxScore)
        
        

    def get_clusters(self, threshold):
        """
        

        Parameters
        ----------
        threshold : int
            all animals in the same group should be above this similarity score

        Returns
        -------
        clusters : 2d list
            list of animals that are part of subtrees that are above the threshold

        """
        if self.root:
            clusters = []
            self.root.get_clusters(threshold, clusters)
        return clusters
            
    
class TreeNode(object):
    def __init__(self, key):
        self.left = None
        self.right = None
        self.key = key
        self.inorder_pos = 0
        
        
    def get_clusters(self, threshold, clusters):
        """
        

        Parameters
        ----------
        threshold : int
            all animals in the same group should be above this similarity score
        clusters : 2d list
            list of animals that are part of subtrees that are above the threshold

        Returns
        -------
        clusters : 2d list
            list of animals that are part of subtrees that are above the threshold

        """
        
        if self.is_leaf():
            #if leaf node, add the animal to clusters list on its own
            clusters.append([self.key])
        else:
            
            #if score is less than threshold, recursively get all leaf nodes of subtree and add to clusters list
            if self.key > threshold:                
                
                clusters.append(self.build_clusters_setup())
                
            #if score is greater than threshold, recurse on both children 
            else:
                if self.left:
                    self.left.get_clusters(threshold, clusters)
                if self.right:
                    self.right.get_clusters(threshold, clusters)
        
        
        return clusters
    
    def build_clusters_setup(self):
        """
        initializes keylist that will hold leaf nodes in subtree

        Returns
        -------
        key_list : list
            completed cluster

        """
        key_list = []
        if self:
            self.build_clusters(key_list)
        return key_list
    
            
    def build_clusters(self, key_list):
        """
        recursively capture all leaf nodes in the subtree        
        
        
        Returns
        -------
        key_list : list
            filling with leaf nodes

        """
        if not self.right and not self.left:
            key_list.append(self.key)
        
        if self.left:
            self.left.build_clusters(key_list)
        if self.right:
            self.right.build_clusters(key_list)
        
        
        return key_list
    
    
    def inorder_setup(self):
        key_list = []
        if self:
            self.inorder([0], key_list)
        return key_list
    
    def inorder(self, num, key_list):
        """
        

        Parameters
        ----------
        num : list
            the in order position of the current node
        key_list : list 
            node keys that are traversed through

        Returns
        -------
        key_list
        
        """
        
        if self.left:
            self.left.inorder(num, key_list)
        self.inorder_pos = num[0]
        key_list.append(self.key)
        num[0] += 1
        if self.right:
            self.right.inorder(num, key_list)
        return key_list
    
    def draw_setup(self, x, maxScore):
        self.inorder_setup()
        if self:
            self.draw(x, maxScore)
            
    def draw(self, x, maxScore):
        """
        recursively draws the dendrogram

        Parameters
        ----------
        x : int
            the x coordinate where the current node will be drawn
            
        maxScore : int
            the x coordinate where the leaves will be drawn

        """
        y = self.inorder_pos
        plt.scatter([x], [y], 25, 'k')
        
        plt.text(x+0.2, y, "{}".format(self.key))
        
        if self.left:
            if self.left.is_leaf():
                x_next = maxScore
            else:
                x_next = self.left.key
            y_next = self.left.inorder_pos
            
            plt.plot([x, x_next], [y, y_next])
            self.left.draw(x_next, maxScore)
        if self.right:
            if self.right.is_leaf():
                x_next = maxScore
            else:
                x_next = self.right.key
            y_next = self.right.inorder_pos
            plt.plot([x, x_next], [y, y_next])
            self.right.draw(x_next, maxScore)
        
        
    def is_leaf(self):
        """
        helper for determining if a node is a leaf

        """
        if type(self.key) is str:
            return True
        return False
        
            
def load_blosum(filename):
    """
    Load in a BLOSUM scoring matrix for Needleman-Wunsch

    Parameters
    ----------
    filename: string
        Path to BLOSUM file
    
    Returns
    -------
    A dictionary of {string: int}
        Key is string, value is score for that particular 
        matching/substitution/deletion
    """
    fin = open(filename)
    lines = [l for l in fin.readlines() if l[0] != "#"]
    fin.close()
    symbols = lines[0].split()
    X = [[int(x) for x in l.split()] for l in lines[1::]]
    X = np.array(X, dtype=int)
    N = X.shape[0]
    costs = {}
    for i in range(N-1):
        for j in range(i, N):
            c = X[i, j]
            if j == N-1:
                costs[symbols[i]] = c
            else:
                costs[symbols[i]+symbols[j]] = c
                costs[symbols[j]+symbols[i]] = c
    return costs


def needleman_wunsch(s1, s2, costs):
    """
    finds the similarity between two amino acid sequences by
    swapping, deleting, or matching characters    
    
    Parameters
    ----------
    s1 : string
        first amino acid sequence
    s2 : string
        second amino acid sequence
    costs : dictionary(
            string (single amino acid): int (cost of swapping, deleting, or matching character)
        )

    Returns
    -------
    cost : int
        The maximum cost of matching the two amino acid sequences
    """
    
    #create M+1 x N+1 table to store solutions
    
    M = len(s1)
    N = len(s2)
    
    
    S = np.zeros((M+1, N+1))
    
    #fill in base cases
    for j in range(1,N+1):
        S[0][j] = S[0][j-1] + costs[(s2[j-1])]
    for i in range(1,M+1):
        S[i][0] = S[i-1][0] + costs[(s1[i-1])]
        
    
    # use previous cases to fill in rest of table
    for i in range(1, M+1):
        for j in range(1, N+1):
            score1 = S[i-1][j-1] + costs[s1[i-1]+s2[j-1]]   
            score2 = S[i][j-1] + costs[s2[j-1]]
            score3 = S[i-1][j] + costs[s1[i-1]]
            S[i][j] = max(score1, score2, score3)
    
    cost = int(S[-1][-1])
    
    return cost

def needleman_all_pairs(keys, species, costs):
    """
    iteratively computes a needleman wunsch score for all pairs of animals and
    stores dictionary in a json    

    Parameters
    ----------
    keys : list
        list of all animals
    species : dictionary(
            string (animal name): string (amino acid sequence)
        )
    costs : dictionary(
            string (single amino acid): int (cost of swapping, deleting, or matching character)
        )

    Returns
    -------
    Creates distances file that stores a needleman wunsch score for each pair of animals

    """
    
    data = {}
    
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            score = needleman_wunsch(species[keys[i]], species[keys[j]], costs)
            data[keys[i] + "," + keys[j]] = score
    
    
    with open("distances.json", "W") as outfile:
        json.dump(data, outfile)
            


costs = load_blosum("blosum62.bla")
species = json.load(open("organisms.json"))
keys = list(species.keys())

# only need to run the following line once
#needleman_all_pairs(keys, species, costs)

data = json.load(open("distances.json"))


T = DendrogramTree()
T.construct_dendrogram(keys, data)



clusters = T.get_clusters(1350)
for group in clusters:
    print(group)


