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

__all__ = ['MTree', 'M_LB_DIST_confirmed', 'M_LB_DIST_non_confirmed', 'generalized_hyperplane']

import abc
from itertools import combinations, islice


def M_LB_DIST_confirmed(entries, current_routing_entry, d):
    """Promotion algorithm. Maximum Lower Bound on DISTance. Confirmed.

    Return the object that is the furthest apart from current_routing_entry 
    using only precomputed distances stored in the entries.
    
    This algorithm does not work if current_routing_entry is None and the
    distance_to_parent in entries are None. This will happen when handling
    the root. In this case the work is delegated to M_LB_DIST_non_confirmed.
    
    arguments:
    entries: set of entries from which two routing objects must be promoted.
    current_routing_entry: the routing_entry that was used
    for the node containing the entries previously.
    None if the node from which the entries come from is the root.
    d: distance function.
    """
    #performance hit of the any?
    if current_routing_entry is None or \
            any(e.distance_to_parent is None for e in entries):
        return M_LB_DIST_non_confirmed(entries,
                                       current_routing_entry,
                                       d)
    
    #if entries contain only one element or two elements twice the same, then
    #the two routing elements returned could be the same. (could that happen?)
    new_entry = max(entries, key=lambda e: e.distance_to_parent)
    return (current_routing_entry.obj, new_entry.obj)

def M_LB_DIST_non_confirmed(entries, unused_current_routing_entry, d):
    """Promotion algorithm. Maximum Lower Bound on DISTance. Non confirmed.

    Compares all pair of objects (in entries) and select the two who are
    the furthest apart.
    """
    objs = map(lambda e: e.obj, entries)
    return max(combinations(objs, 2), key=lambda (x, y): d(x, y))

#If the routing objects are not in entries it is possible that
#all the elements are in one set and the other set is empty.
def generalized_hyperplane(entries, routing_object1, routing_object2, d):
    """Partition algorithm.

    Each entry is assigned to the routing_object to which it is the closest.
    This is an unbalanced partition strategy.

    Return a tuple of two elements. The first one is the set of entries
    assigned to the routing_object1 while the second is the set of entries
    assigned to the routing_object2"""
    partition = (set(), set())
    for entry in  entries:
        partition[d(entry.obj, routing_object1) > \
                         d(entry.obj, routing_object2)].add(entry)    
    return partition


#TODO: node size : 32 is arbitrary. Define a reasonable default value
#make a few tests and count number of call to d (hint: decorate d)
class MTree(object):
    def __init__(self,
                 d,
                 max_node_size=32,
                 promote=M_LB_DIST_confirmed,
                 partition=generalized_hyperplane):
        """
        Create a new MTree.

        Arguments:
        d: distance function.
        max_node_size: optional. Maximum number of entries in a node of
            the M-tree
        promote: optional. Used during insertion of a new value when
            a node of the tree is split in two.
            Determines given the set of entries which two entries should be
            used as routing object to represent the two nodes in the
            parent node.
            This is delving pretty far into implementation choices of
            the Mtree. If you don't understand all this just swallow
            the blue pill, use the default value and you'll be fine.
        partition: optional. Used during insertion of a new value when
            a node of the tree is split in two.
            Determines to which of the two routing object each entry of the
            split node should go.
            This is delving pretty far into implementation choices of
            the Mtree. If you don't understand all this just swallow
            the blue pill, use the default value and you'll be fine.
        """
        if not callable(d):
            #Why the hell did I put this?
            #This is python, we use dynamic typing and assumes the user
            #of the API is smart enough to pass the right parameters.
            raise TypeError('d is not a function')
        if max_node_size < 2:
            raise ValueError('max_node_size must be >= 2 but is %d' %
                             max_node_size)
        self.d = d
        self.max_node_size = max_node_size
        self.promote = promote
        self.partition = partition
        self.size = 0
        self.root = LeafNode(d, self)

    def __len__(self):
        return self.size

    def add(self, obj):
        """
        Add an object into the M-tree
        """
        self.root.add(obj)
        self.size += 1

    def add_all(self, iterable):
        """
        Add all the elements in the M-tree
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

    def __repr__(self):
        return "<Entry obj: %r, dist: %r, radius: %r, subtree: %r>" % (
            self.obj,
            self.distance_to_parent,
            self.radius,
            self.subtree.repr_class() if self.subtree else self.subtree)


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
                 entries=None):
        #There will be an empty node (entries set is empty) when the tree
        #is empty and there only is an empty root.
        #May also be empty during construction (create empty node then add
        #the entries later).
        self.d = d
        self.mtree = mtree
        self.parent_node = parent_node
        self.parent_entry = parent_entry
        self.entries = set(entries) if entries else set()

    def __repr__(self):
        #entries might be big. Only prints the first few elements
        entries_str = '%s' % list(islice(self.entries, 2))
        if len(self.entries) > 2:
            entries_str = entries_str[:-1] + ', ...]'
            
        return "<%s parent_node: %s, parent_entry: %s, entries:%s>" % (
            self.__class__.__name__,
            self.parent_node.repr_class() \
                if self.parent_node else self.parent_node,
            self.parent_entry,
            entries_str
            
    )

    def repr_class(self):
        return "<" + self.__class__.__name__ + ">"

    def __len__(self):
        return len(self.entries)

    def is_full(self):
        return len(self) == self.mtree.max_node_size

    def is_empty(self):
        return len(self) == 0

    def is_root(self):
        return self is self.mtree.root

    def remove_entry(self, entry):
        """Removes the entry from this node

        Raise KeyError if the entry is not in this node
        """
        self.entries.remove(entry)

    def add_entry(self, entry):
        """Add an entry to this node.

        Raise ValueError if the node is full.
        """
        if self.is_full():
            raise ValueError('Trying to add %s into a full node' % str(entry))
        self.entries.add(entry)

    @abc.abstractmethod
    def add(self, obj):
        """Add obj into this subtree"""
        pass

    @abc.abstractmethod
    def covering_radius_for(self, obj):
        """Compute the radius needed for obj to cover the entries of this node.
        """
        pass
        

class LeafNode(AbstractNode):
    """A leaf of the M-tree"""
    def __init__(self,
                 d,
                 mtree,
                 parent_node=None,
                 parent_entry=None,
                 entries=None):

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
            split(self, new_entry, self.d)
        assert self.is_root() or self.parent_node

    def covering_radius_for(self, obj):
        """Compute minimal radius for obj so that it covers all the objects
        of this node.
        """
        if not self.entries:
            return 0
        else:
            return max(map(lambda e: self.d(obj, e.obj), self.entries))
        
    
class InternalNode(AbstractNode):
    """An internal node of the M-tree"""

    def __init__(self,
                 d,
                 mtree,
                 parent_node=None,
                 parent_entry=None,
                 entries=None):

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
            dist_to_obj[e] = self.d(obj, e.obj)

        def find_best_entry_requiring_no_covering_radius_increase():
            valid_entries = filter(lambda e : dist_to_obj[e] <= e.radius,
                                   self.entries)
            
            return min(valid_entries, key=dist_to_obj.get) \
                if valid_entries else None
                
        def find_best_entry_minimizing_radius_increase():
            entry = min(self.entries,
                             key=lambda e: dist_to_obj[e] - e.radius)
            #enlarge radius so that obj is in the covering radius of e 
            entry.radius = dist_to_obj[entry]
            return entry

        entry = find_best_entry_requiring_no_covering_radius_increase() or \
            find_best_entry_minimizing_radius_increase()
        entry.subtree.add(obj)
        assert self.is_root() or self.parent_node

    def covering_radius_for(self, obj):
        """Compute minimal radius for obj so that it covers the radiuses
        of all the routing objects of this node
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
    Split existing_node into two nodes.

    Adding entry to existing_node causes an overflow. Therefore we
    split existing_node into two nodes.
    
    Arguments:
    existing_node: full node to which entry should have been added
    entry: the added node. Caller must ensures that entry is initialized
           correctly as it would be if it were an effective entry of the node.
           This means that distance_to_parent must possess the appropriate
           value (the ditance to existing_node.parent_entry).
    d: distance function.
    """
    mtree = existing_node.mtree
    #type of the new node must be the same as existing_node
    #parent node, parent entry and entries are set later
    new_node = type(existing_node)(existing_node.d,
                                   existing_node.mtree)
    all_entries = existing_node.entries | set((entry,))

    #It is guaranteed that the current routing entry of the split node
    #(i.e. existing_node.parent_entry) is the one distance_to_parent
    #refers to in the entries (including the entry parameter). 
    #Promote can therefore use distance_to_parent of the entries.
    routing_object1, routing_object2 = \
        mtree.promote(all_entries, existing_node.parent_entry, d)
    entries1, entries2 = mtree.partition(all_entries,
                                         routing_object1,
                                         routing_object2,
                                         d)
    assert entries1 and entries2, "Error during split operation. All the entries have been assigned to one routing_objects and none to the other! Should never happen since at least the routing objects are assigned to there corresponding set  of entries"

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

    def update_entries_parent_node(node):
        for entry in node.entries:
            if entry.subtree:
                entry.subtree.parent_node = node
    update_entries_parent_node(existing_node)
    update_entries_parent_node(new_node)

    #must save the old entry of the existing node because it will have
    #to be removed from the parent node later
    old_existing_node_parent_entry = existing_node.parent_entry
    
    existing_node_entry = build_entry(existing_node, routing_object1)
    existing_node.parent_entry = existing_node_entry

    new_node_entry = build_entry(new_node, routing_object2)
    new_node.parent_entry = new_node_entry

        
    if existing_node.is_root():
        new_root_node = InternalNode(existing_node.d,
                                existing_node.mtree)

        existing_node.parent_node = new_root_node
        new_root_node.add_entry(existing_node_entry)
        
        new_node.parent_node = new_root_node
        new_root_node.add_entry(new_node_entry)
        
        mtree.root = new_root_node
    else:
        parent_node = existing_node.parent_node

        if not parent_node.is_root():
            #parent node has itself a parent, therefore the two entries we add
            #in the parent must have distance_to_parent set appropriately
            existing_node_entry.distance_to_parent = \
                d(existing_node_entry.obj, parent_node.parent_entry.obj)
            new_node_entry.distance_to_parent = \
                d(new_node_entry.obj, parent_node.parent_entry.obj)

        parent_node.remove_entry(old_existing_node_parent_entry)
        parent_node.add_entry(existing_node_entry)
        
        if parent_node.is_full():
            split(parent_node, new_node_entry, d)
        else:
            parent_node.add_entry(new_node_entry)
            new_node.parent_node = parent_node
    assert existing_node.is_root() or existing_node.parent_node
    assert new_node.is_root() or new_node.parent_node
        

def build_entry(node, routing_object, distance_to_parent=None):
    """Return a new entry whose covering tree is node and
    the routing object is routing_object
    """
    covering_radius = node.covering_radius_for(routing_object)
    return Entry(routing_object,
                 distance_to_parent,
                 covering_radius,
                 node)

if __name__ == '__main__':
    max_sizes = range(2, 20) + [1000]
    for max_size in max_sizes:
        tree = MTree(lambda i1, i2: abs(i1 - i2), max_size)
        objs = range(5000)
        for o in objs:
            tree.add(o)
