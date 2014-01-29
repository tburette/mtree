#TODO: doc say that it is (similar to B-tree and) in memory only
#TODO: doc specify exactly what d can return :
# (something that work like) a number
# value always the same for same param
# + properties
#TODO: doc specify what exactly an obj can be
# anything that can be used by the d provided.
# never do anything with obj except store it and use it as param to d
#TODO: doc talk about duplicate obj (same object or object with d(x, y) = 0).
# Do not check and allow duplicates in the tree.
#  verify it is true + say so in docstring
#TODO: doc usage and example
"""Search for elements that are the most similar to a given one

The M-tree is a data structure that can store elements and search for them. The particularity is that instead of performing exact search to find elements that match exactly the search query, it performs similarity queries, that is finding the elements that are the most similar to a search query.

The M-tree is a tree based implementation of a metric space ( http://en.wikipedia.org/wiki/Metric_space ).


Usage:
#TODO: cf Lib/heapq.py

Example:
#TODO: simple example using strings
file:///Users/burettethomas/Documents/dev/python/python-2.7.2-docs-html/library/doctest.html


Implementation based on the paper
'M-tree: An Efficient Access Method for Similarity Search in Metric Spaces'.

"""
#infinity? (cf k-NN search)
#add a nearest neighbor function (implemented in term of k-NN)
#little tool to check that d is valid

__all__ = ['MTree', 'M_LB_DIST_confirmed', 'generalized_hyperplane']

import abc
from itertools import combinations


def M_LB_DIST_confirmed(entries, current_routing_entry, d):
    """Promotion algorithm. Maximum Lower Bound on DISTance. Confirmed.

    This algorithm does not work if current_routing_entry is None and the
    distance_to_parent in entries is None. This will happen when handling
    the root. In this case the work is delegated to 
    
    arguments:
    entries: set of entries from which two routing objects must be promoted.
    current_routing_entry: the routing_entry that was used
    for the node containing the entries.
    None if the node from which the entries come from is the root.
    d: distance function.
    """
    if current_routing_entry is None or entries[0].distance_to_parent is None:
        return M_LB_DIST_non_confirmed(entries,
                                       current_routing_entry,
                                       d)
    
    #if entries contain only one element or two elements twice the same, then
    #the two routing elements returned could be the same. (could that happen?)
    new_entry = max(entries, key=lambda e: e.distance_to_parent)
    return (current_routing_entry, new_entry)

def M_LB_DIST_non_confirmed(entries, current_routing_entry, d):
    """Promotion algorithm. Maximum Lower Bound on DISTance. Unconfirmed

    Compares all pair of entries and select the two who are the furthest apart.
    """
    return max(combinations(entries, 2), key=lambda (x, y): d(x, y))
        
    

#warning. One set might contain only one element.
#If the routing objects are not in entries it might even be possible that
#all the elements are in one set.
def generalized_hyperplane(entries, routing_object1, routing_oject2, d):
    """Partition algorithm

    Returns a tuple of two elements. The first one is the set of entries
    assigned two the routing_object1 while the second is the set of entries
    assigned to routing_object2"""
    entries_list = (set(), set())
    for entry in  entries:
        entries_list[d(entry.obj, routing_object1) > \
                         d(entry.obj, routing_object2)].add(entry)    
    return new_entries


#TODO: node size : 32 is arbitrary. Define a reasonable default value
class MTree(object):
    def __init__(self,
                 d,
                 max_node_size=32,
                 promote=M_LB_DIST_confirmed,
                 partition=generalized_hyperplane):
        """
        Creates a new MTree.

        Arguments:
        d: distance function.
        max_node_size: optional. Maximum number of entries in a node of
            the M-tree
        promote: optional. Used during insertion when a node is split in two.
            Determines given the set of entries which two entries should be
            used as routing object to represent the two nodes in the
            parent node.
            This is delving pretty far into implementation alternatives of
            the Mtree. If you don't understand what this all means just use the
            default value and you'll be fine.
        partition: optional. Used during insertion when a node is split in two.
            Determines in which of the two node each entry should go.
            This is delving pretty far into implementation alternatives of
            the Mtree. If you don't understand what this all means just use the
            default value and you'll be fine.
        """
        if not callable(d):
            raise TypeError('d is not a function')
        if max_node_size < 2:
            raise ValueError('max_node_size must be >= 2 but is %d' %
                             max_node_size)
        self.size = 0
        self.d = d
        self.max_node_size = max_node_size
        self.promote = promote
        self.partition = partition
        self.root = LeafNode(d, self)

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


class AbstractNode(object):
    """An abstract leaf of the M-tree.

    Concrete class are LeafNode and InternalNode

    We need to keep a reference to mtree so that we can know if a given node
    is root as well as update the root.
    
    We need to keep both the parent entry and the parent node (i.e. the node
    in which the parent entry is) for the split operation. During a split
    we may need to remove the parent entry from the node as well as adding
    a new entry to the node."""

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 d,
                 mtree,
                 parent_node=None,
                 parent_entry=None,
                 entries=set()):
        #A node is empty (no entry) when the tree is empty.
        #May also be empty during construction (create empty node then add
        #the values).
        self.d = d
        self.mtree = mtree
        self.parent_node = parent_node
        self.parent_entry = parent_entry
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def is_full(self):
        return len(self) == self.mtree.max_node_size

    def is_empty(self):
        return len(self) == 0

    def is_root(self):
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
        if self.is_full():
            raise TypeError('Trying to add %s into a full node' % str(entry))
        self.entries.add(self, entry)

    @abc.abstractmethod
    def add(self, obj):
        pass

    @abc.abstractmethod
    def covering_radius_for(self, obj):
        pass
        

class LeafNode(AbstractNode):
    """A leaf of the M-tree"""
    def __init__(self,
                 d,
                 mtree,
                 parent_node=None,
                 parent_entry=None,
                 entries=set()):

        AbstractNode.__init__(self,
                              d,
                              mtree,
                              parent_node,
                              parent_entry,
                              entries)
    def add(self, obj):
        distance_to_parent = self.d(obj, self.parent_entry.obj) \
            if self.parent_entry else None
        new_entry = Entry(obj, distance_to_parent)
        if not self.is_full():
            self.entries.add(new_entry)
        else:
            split(self, new_entry, d)

    def covering_radius_for(self, obj):
        """Compute minimal radius for obj so that it covers all the objects
        of this node.
        """
        if not self.entries:
            return 0
        else:
            return max(map(lambda e: self.d(obj, e.obj), self.entries))
        
    
class InternalNode(object):
    """An internal node of the M-tree"""
    def __init__(self,
                 d,
                 mtree,
                 parent_node=None,
                 parent_entry=None,
                 entries=set()):

        AbstractNode.__init__(self,
                              d,
                             mtree,
                              parent_node,
                              parent_entry,
                              entries)

    #TODO: apply optimization that uses the d of the parent to reduce the
    #number of d computation performed. cf M-Tree paper 3.3
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
        if not self.entries:
            return 0
        else:
            return max(map(lambda e: self.d(obj, e.obj) + e.radius,
                           self.entries))

#A lot of the code is duplicated to do the same operation on the existing_node
#and the new node :(. Could prevent that by creating a set of two elements and
#perform on the (two) elements of that set.
def split(existing_node, entry, d):
    """
    splits the node into two nodes.
    
    Arguments:
    existing_node: full node to which entry should have been added
    entry: the added node
    """
    mtree = existing_node.mtree
    #type of the new node must be the same as existing_node
    #parent node, parent entry and entries are set later
    new_node = type(existing_node)(existing_node.d,
                                   existing_node.mtree)
    all_entries = existing_node.entries & set((entry,))

    #It is guaranteed that the current routing entry of the split node
    #(i.e. existing_node.parent_entry) is the one distance_to_parent
    # refers to in the entries (including the entry parameter).
    routing_object1, routing_object2 = \
        mtree.promote(all_entries, existing_node.parent_entry, d)
    entries1, entries2 = mtree.partition(all_entries,
                                         routing_object1,
                                         routing_object2,
                                         d)
    assert not entries1 or not entries2, "Error during split operation. All the entries have been assigned to one routing_objects and none to the other! Should never happen since at least the entry corresponding to the routing_object should be assigned to it."

    existing_node.entries = entries1
    new_node.entries = entries2

    #wastes d computations if parent hasn't changed.
    #How to avoid? -> check if routing_object1/2 is the same as the old routing
    
    #promote/partition probably did similar d computations.
    #How to avoid recomputations between promote, partition and this?
    #share cache (a dict) passed between functions?
    #memoization? (with LRU!).
    #    id to order then put the two objs in a tuple (or rather when fetching
    #      try both way
    #    add a function to add value without computing them
    #      (to add distance_to_parent)
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
        
    if existing_node.is_root():
        new_root_node = InternalNode(existing_node.d,
                                existing_node.mtree)

        existing_node.parent_node = new_root
        new_root.add_entry(existing_node_entry)
        
        new_node.parent_node = new_root
        new_root.add_entry(new_node_entry)
        
        mtree.root = new_root
    else:
        parent_node = existing_node.parent_node

        if not parent_node.is_root():
            #parent node has itself a parent, therefore the two entries we add
            #in the parent must have distance_to_parent set appropriately
            existing_node_entry.distance_to_parent = \
                d(existing_node_entry.obj, parent_node.parent_entry.obj)
            new_node_entry.distance_to_parent = \
                d(new_node_entry.obj, parent_node.parent_entry.obj)

        parent_node.remove_entry(old_existing_existing_node)
        parent_node.add_entry(existing_node_entry)
        
        if parent_node.is_full():
            split(parent_node, new_node_entry, d)
        else:
            parent_node.add_entry(new_node_entry)
            new_node.parent_node = parent_node
        

def build_entry(node, routing_object, distance_to_parent=None):
    """Returns a new entry whose covering tree is node and
    the routing object is routing_object
    """
    covering_radius = node.covering_radius_for(routing_object)
    return Entry(routing_object,
                 distance_to_parent,
                 covering_radius,
                 node)    
