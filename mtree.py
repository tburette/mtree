#TODO: add: appliquer optimisation qui utilise d du parent pour reduire d faits
#infinity? (cf k-NN search)
#implement only one promote and partition policies but code so that
# other can be added later
#split: handle case where a node had no parent but now has one.
# Must compute and assign d(Oj, Op)
#TODO: specify in docstring exactly what d can return :
# (something that work like) a number
#TODO: specify in docstring what exactly an obj can be
# anything that can be used by the d provided.
# never do anything with obj except store it and use it as param to d
#talk about duplicates in doc (same object or object with d(x, y) = 0).
# Do not check and allow duplicates in the tree.
#  verify it is true + say so in docstring
#add a nearest neighbor function (that is implemented in term of k-NN)
#entries of a node should never be empty(right?) so remove default value
# of none for entries parameters

__all__ = ['MTree']

#TODO: node size : 32 is arbitrary. Define a reasonable default value
class MTree():
    def __init__(self, d, max_node_size=32):
        """
        Creates a new MTree.

        d: metric of the metric space.
        max_node_size: optional. Maximum number of entries in a node of
        the M-tree
        """
        self.size = 0
        self.root = Leaf(d, max_node_size)
        self.d = d

    def __len__(self):
        return self.size

    def add(self, obj):
        """
        Adds an object into the M-tree
        """
        self.root.add(obj)
        self.size += 1


class LeafEntry():
    """
    The leafs of the M-tree contain a list of instances of this class.
    Each instance represents an object that is stored by this tree

    The distance to the parent is None if the leaf in which this entry is
    stored has no parent"""
    def __init__(self, obj, distance_to_parent=None):
        self.obj = obj
        self.distance_to_parent = distance_to_parent

class InternalNodeEntry():
    """
    The internal nodes of the M-tree contain a list of instances of this class.
    Each instance represents a routing object.

    The distance to the parent is None if the leaf in which this entry is
    stored has no parent.
    """
    def __init__(self,
                 obj,
                 radius,
                 distance_to_parent=None,
#TODO: shouldn't an internal node always have a covering tree?
                 subtree=None):
        self.obj = obj
        self.radius = radius
        self.distance_to_parent = distance_to_parent
        self.subtree = subtree
        
    
class Leaf(Node):
    """A leaf of the M-tree"""
    def __init__(self, d, max_node_size, parent = None, entries=set()):
        self.d = d
        self.max_node_size = max_node_size
        self.parent = parent
        self.entries = entries

    def __len__(self):
        return len(entries)

    def add(self, obj):
        distance_to_parent = self.d(obj, parent) if parent else None
        new_entry = LeafEntry(obj, distance_to_parent)
        if(len(self) < self.max_node_size):
            entries.add(new_entry)
        else:
            split(self, new_entry)

    #move outside the Node class? module level method?
    def split(self, new_entry):
        raise Exception('Not yet implemented')
                                  
            
            

    
class InternalNode(Node):
    """An internal node of the M-tree"""
    def __init__(self, d, max_node_size, parent = None, entries=set()):
        self.d = dg
        self.max_node_size = max_node_size
        self.parent = parent
        self.entries = entries

    def __len__(self):
        return len(entries)
        
    def add(self, obj):     
        #put d(obj, e) in a dict to prevent recomputation 
        #I guess memoization could be used to make code clearer but that is
        #too magic for me plus there is potentially a very large number of
        #calls to memoize
        dist_to_obj = {}
        for e in self.entries:
            #Use Entry as key to dict even though
            #eq and hash haven't been defined
            dist_to_obj[e] = d(obj, e)

        def find_best_entry_requiring_no_covering_radius_increase():
            valid_entries = filter(lambda e : dist_to_obj[e] <= e.radius,
                                   self.entries)
            if valid_entries:
                return min(valid_entries, key=dist_to_obj.get)
            else: return None
                
        def find_best_entry_minimizing_radius_increase():
            entry = min(self.entries,
                             key=lambda e: dist_to_obj[e] - e.radius)
            #enlarge radius so that obj is in the covering radius of e 
            entry.radius = dist_to_obj[e]
            return entry

        entry = find_best_entry_requiring_no_covering_radius_increase() or \
            find_best_entry_minimizing_radius_increase()
        entry.add(obj)
