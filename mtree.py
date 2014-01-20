#TODO: code duplication between Leaf and InternalNode. Create common base class
#TODO: say that is (similar to B-tree and) in memory only
#TODO: specify in docstring exactly what d can return :
# (something that work like) a number
#TODO: specify in docstring what exactly an obj can be
# anything that can be used by the d provided.
# never do anything with obj except store it and use it as param to d
#talk about duplicates in doc (same object or object with d(x, y) = 0).
# Do not check and allow duplicates in the tree.
#  verify it is true + say so in docstring
"""Search for elements that are the most similar to a given one

The M-tree is a data structure that can store elements and search for them. The particularity is that instead of performing exact search to find elements that match exactly the search query, it performs similarity queries, that is finding the elements that are the most similar to a search query.

The M-tree is a tree based implementation of a metric space ( http://en.wikipedia.org/wiki/Metric_space ).


Usage:
#TODO: cf Lib/heapq.py

Example:
#TODO: simple example using strings


Implementation based on the paper
'M-tree: An Efficient Access Method for Similarity Search in Metric Spaces'.

"""
#infinity? (cf k-NN search)
#add a nearest neighbor function (that is implemented in term of k-NN)
#entries of a node should never be empty(right?) so remove default value
# of none for entries __init__ parameters
# nope : empty tree

__all__ = ['MTree']

#TODO: node size : 32 is arbitrary. Define a reasonable default value
class MTree(object):
    def __init__(self, d, max_node_size=32):
        """
        Creates a new MTree.

        d: distance function.
        max_node_size: optional. Maximum number of entries in a node of
        the M-tree
        """
        self.size = 0
        self.root = LeafNode(d, max_node_size, self)
        self.d = d

    def __len__(self):
        return self.size

    def add(self, obj):
        """
        Adds an object into the M-tree
        """
        self.root.add(obj)
        self.size += 1

    def add_all(self, iterable):
        """
        Adds all the elements in the M-tree
        """
        #TODO: implement using the bulk-loading algorithm
        for x in iterable:
            self.add(x)


class Entry(object):
    """
    
    The leafs and internal nodes the M-tree contain a list of instances of this
    class.

    The distance to the parent is None if the leaf in which this entry is
    stored has no parent.

    radius and subtree are None if the entry is contained in a leaf.

    Used in set and dict even tough eq and hash haven't been redefined
    """
    def __init__(self,
                 obj,
                 distance_to_parent=None,
                 radius=None,
                 subtree=None):
        self.obj = obj
        self.distance_to_parent = distance_to_parent
        self.radius = radius
        self.subtree = subtree


class LeafNode(object):
    """A leaf of the M-tree

    We need to keep a reference to mtree so that we can know if a given node
    is root as well as update the root.
    
    We need to keep both the parent entry and the parent node (i.e. the node
    in which the parent entry is) for the split operation. During a split
    we may need to remove the parent entry from the node as well as adding
    a new entry to the node. (This is also the case for InternalNode)"""
    def __init__(self,
                 d,
                 max_node_size,
                 mtree,
                 parent_node=None,
                 parent_entry=None,
                 entries=set()):
        self.d = d
        self.max_node_size = max_node_size
        self.mtree = mtree
        self.parent_node = parent_node
        self.parent_entry = parent_entry
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def isfull(self):
        return len(self) == self.max_node_size

    def isroot(self):
        return self is mtree.root

    def remove_entry(self, entry):
        """ remove the entry from this node

        Raises KeyError if the entry is not in this node
        """
        self.entries.remove(entry)

    def add_entry(self, entry):
        """Adds an entry to this node.

        Raises TypeError if the node is full.
        """
        if self.isfull():
            raise TypeError('Trying to add %s into a full node' % str(entry))
        self.entries.add(self, entry)

    def add(self, obj):
        distance_to_parent = self.d(obj, self.parent_entry.obj) \
            if self.parent_entry else None
        new_entry = Entry(obj, distance_to_parent)
        if not self.isfull():
            self.entries.add(new_entry)
        else:
            split(self, new_entry, d)

    def covering_radius_for(self, obj):
        """Compute minimal radius for obj so that it covers all the objects
        of this node.
        """
        return max(map(lambda e: self.d(obj, e.obj), self.entries))
        
    
class InternalNode(object):
    """An internal node of the M-tree"""
    def __init__(self,
                 d,
                 max_node_size,
                 mtree,
                 parent_node = None,
                 parent_entry = None,
                 entries=set()):
        self.d = d
        self.max_node_size = max_node_size
        self.mtree = mtree
        self.parent_node = parent_node
        self.parent_entry = parent_entry
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def isfull(self):
        return len(self) == self.max_node_size

    def isroot(self):
        return self is mtree.root

    def remove_entry(self, entry):
        self.entries.remove(entry)

    def add_entry(self, entry):
        if self.isfull():
            raise TypeError('Trying to add %s into a full node' % str(entry))
        self.entries.add(self, entry)
    
    #TODO: appliquer optimisation qui utilise d du parent pour reduire d faits
    # cf M-Tree paper 3.3
    def add(self, obj):     
        #put d(obj, e) in a dict to prevent recomputation 
        #I guess memoization could be used to make code clearer but that is
        #too magic for me plus there is potentially a very large number of
        #calls to memoize
        dist_to_obj = {}
        for e in self.entries:
            dist_to_obj[e] = d(obj, e)

        def find_best_entry_requiring_no_covering_radius_increase():
            valid_entries = filter(lambda e : dist_to_obj[e] <= e.radius,
                                   self.entries)
            
            min(valid_entries, key=dist_to_obj.get) if valid_entries else None
                
        def find_best_entry_minimizing_radius_increase():
            entry = min(self.entries,
                             key=lambda e: dist_to_obj[e] - e.radius)
            #enlarge radius so that obj is in the covering radius of e 
            entry.radius = dist_to_obj[e]
            return entry

        entry = find_best_entry_requiring_no_covering_radius_increase() or \
            find_best_entry_minimizing_radius_increase()
        entry.add(obj)

    def covering_radius_for(self, obj):
        """Compute minimal radius for obj so that it covers all the radius
        of the routing objects of this node
        """
        return max(map(lambda e: self.d(obj, e.obj) + e.radius, self.entries))

                                  
def split(existing_node, entry, d):
    """
    splits the node into two nodes.
    
    Arguments:
    existing_node: full node to which entry should have been added
    entry: the added node
    """
    #type of the new node must be the same as existing_node
    #parent node, parent entry and entries are set later
    new_node = type(existing_node)(existing_node.d,
                                   existing_node.max_node_size,
                                   existing_node.mtree,
                                   None,
                                   None,
                                   None)
    all_entries = existing_node.entries & set((entry,))

    routing_object1, routing_object2 = promote(all_entries)
    entries1, entries2 = partition(all_entries,
                                   routing_object1,
                                   routing_object2)

    existing_node.entries = entries1
    new_node.entries = entries2
    def update_entries_distance_to_parent(entries, parent_routing_object):
        for entry in entries:
            entry.distance_to_parent = d(entry.obj, parent_routing_object)
    update_entries_distance_to_parent(existing_node.entries, routing_object1)
    update_entries_distance_to_parent(new_node.entries, routing_object2)

    #must save the old entry of the existing node because it will have
    #to be removed from the parent node later
    old_existing_node_entry = existing_node.parent_entry
    existing_node_entry = build_entry(existing_node, routing_object1)
    existing_node.parent_entry = existing_node_entry

    new_node_entry = build_entry(new_node, routing_object2)
    new_node.parent_entry = new_node_entry    
        
    if existing_node.isroot():
        new_root_node = InternalNode(existing_node.d,
                                existing_node.max_node_size,
                                existing_node.mtree)

        existing_node.parent_node = new_root
        new_root.add_entry(existing_node_entry)
        
        new_node.parent_node = new_root
        new_root.add_entry(new_node_entry)
        
        mtree.root = new_root
    else:
        #!! cas ou parent est full : que contient l'entry?
        parent_node = existing_node.parent_node

        if not parent_node.isroot():
            #parent node has itself a parent, therefore the entries of
            #parent node must have distance_to_parent set appropriately
            existing_node_entry.distance_to_parent = \
                d(existing_node_entry.obj, parent_node.parent_entry.obj)
            new_node_entry.distance_to_parent = \
                d(new_node_entry.obj, parent_node.parent_entry.obj)

        parent_node.remove_entry(old_existing_existing_node)
        parent_node.add_entry(existing_node_entry)
        
        if parent_node.isfull():
            split(parent_node, new_node_entry, d)
        else:
            parent_node.add_entry(new_node_entry)
            new_node.parent_node = parent_node
        

def build_entry(node, routing_object, distance_to_parent=None):
    """Returns a new entry whose covering tree is node and
    the routing_object is the parameter.
    """
    covering_radius = node.covering_radius_for(routing_object)
    return Entry(routing_object,
                 distance_to_parent,
                 covering_radius,
                 node)
    
#should be passed to MTree with default value
def promote(entries):
    pass

def partition(entries, routing_entry1, routing_entry2):
    pass
