"""M-tree datastructure to perform k-NN searches
===
This is an implementation of the M-tree, a data structure to find the element(s) the most similar to a given one.

The M-tree is a tree based implementation of the concept of metric space
( http://en.wikipedia.org/wiki/Metric_space ), it is similar to b-tree.

Implementation based on the paper
'M-tree: An Efficient Access Method for Similarity Search in Metric Spaces'

To use the M-tree you only need to pass two things to it:

- a set of objects to store.
- a distance function `d(x, y)` that returns a number establishing
how similar two objects are.

Usage:
======
    >>> def d_int(x, y):      # define a distance function for numbers
    ...     return abs(x - y)
    ...
    >>> tree = MTree(d_int, max_node_size=4)   # create an empty M-tree
    >>> tree.add(1)           # add objects 1, 5 and 9 to the tree
    >>> tree.add_all([5, 9])
    >>> tree.search(10)       # search the object closest to 10. Will return 9
    >>> [9]
    >>> tree.search(9, 2)     # search the two objects closest to 9.
    >>> [5, 9]

The size of nodes (optional argument `max_node_size`) has a large influence on 
the number of calls of the distance function (`d`).

The objects you insert in the tree can be anything as long as the
distance function you provide is able to handle them corretly.

The distance function (`d`) must be provided when the tree is created.
It takes as a parameter two objects and return an number telling how
similar the two objects are. The smaller the number, the more similar the
objects are. The number returned can be an integer, float,... Technically
anything that behaves like a number (<, <=, >,... work like numbers).

The distance function MUST respect the following properties:

- d always return the same value given the same parameters
- Non negativity: forall x, y: d(x, y) >= 0
  d must never return a negative value. If the value your function returns
  can be negative but has a lower bound (e.g. never returns anything lower
  than -100) you can fix this by systematically increasing the value of
  all the number returned (e.g. return value +100).
- Symmetry: forall x, y: d(x, y) = d(y, x)
  The same value must be returned no matter what the order of the parameters
  are.
- Identity: forall x, y: d(x, y) = 0 means that x = y
- Triangle inequality: forall x, y, z d(x, z) <= d(x, y) + d(y, z)
  The distance from one point to a second is always smaller or equal to the
  the distance from one point to an intermediary + the distance from the
  intermediary to the second point.
  Here is an analogy to help understand this property. Imagine a road
  going directlty between two towns. It never turns, it is a perfectly
  straight line. This is obviously the shortest way to get between the two
  town. Now imagine we pick a position anywhere we want. If we go from
  one town to the other by passing trough this position, it is impossible to
  have travelled less than by following the straight road.

If the distance function violates one of these rule, the M-tree may
return erroneous results. 

If the same object is inserted multiple times, it will be considered as
different object by the tree.

This implementation is memory only. The tree is not stored on disk.
This may be a problem if the objects you store are large (pictures, sound,...)
Altough the tree itself resides in memory you can store the objects it contains on disk (or online,...). For example the objects you pass to the tree could
be path to files; the d function would load the files from disk to perform the
comparisons.

To maintain good performance while minimizing memory usage, a good trade-off
is to store in the objects only the path to the actual data as well as the key
features that define the data. The distance function (d) can then compare
the objects using the features without the need for disk access
That way, searches are fast (no disk access) while keeping data on disk.

"""

__all__ = ['MTree', 'M_LB_DIST_confirmed', 'M_LB_DIST_non_confirmed',
           'generalized_hyperplane']

__version__ = '1.0.0'

import abc
from heapq import heappush, heappop
import collections
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
    return max(combinations(objs, 2), key=lambda two_objs: d(*two_objs))

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
    for entry in entries:
        partition[d(entry.obj, routing_object1) > \
                         d(entry.obj, routing_object2)].add(entry)

    if not partition[0] or not partition[1]:
        #all objects have been put on the same routing_object
        #can only happen if all objects are the same point (d() always 0)
        #fix by splitting the group in two
        partition = (set(islice(entries, len(entries)//2)),
                     set(islice(entries, len(entries)//2, len(entries))))

    return partition


class MTree(object):
    def __init__(self,
                 d,
                 max_node_size=4,
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
        self.root = LeafNode(self)

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
        for obj in iterable:
            self.add(obj)

    def search(self, query_obj, k=1):
        """Return the k objects the most similar to query_obj.

        Implementation of the k-Nearest Neighbor algorithm.
        Returns a list of the k closest elements to query_obj, ordered by
        distance to query_obj (from closest to furthest).
        If the tree has less objects than k, it will return all the
        elements of the tree."""
        k = min(k, len(self))
        if k == 0: return []

        #priority queue of subtrees not yet explored ordered by dmin
        pr = []
        heappush(pr, PrEntry(self.root, 0, 0))

        #at the end will contain the results 
        nn = NN(k)

        while pr:
            prEntry = heappop(pr)
            if(prEntry.dmin > nn.search_radius()):
                #best candidate is too far, we won't have better a answer
                #we can stop
                break
            prEntry.tree.search(query_obj, pr, nn, prEntry.d_query)

            #could prune pr here
            #(the paper prunes after each entry insertion, instead whe could
            #prune once after handling all the entries of a node)
            
        return nn.result_list()

    
NNEntry = collections.namedtuple('NNEntry', 'obj dmax')
class NN(object):
    def __init__(self, size):
        self.elems = [NNEntry(None, float("inf"))] * size
        #store dmax in NN as described by the paper
        #but it would be more logical to store it separately
        self.dmax = float("inf")

    def __len__(self):
        return len(self.elems)

    def search_radius(self):
        """The search radius of the knn search algorithm.
        aka dmax

        The search radius is dynamic."""
        return self.dmax

    def update(self, obj, dmax):
        if obj == None:
            #internal node
            self.dmax = min(self.dmax, dmax)
            return
        self.elems.append(NNEntry(obj, dmax))
        for i in range(len(self)-1, 0, -1):
            if self.elems[i].dmax < self.elems[i-1].dmax:
                self.elems[i-1], self.elems[i] = self.elems[i], self.elems[i-1]
            else:
                break
        self.elems.pop()

    def result_list(self):
        result = map(lambda entry: entry.obj, self.elems)
        return result

    def __repr__(self):
        return "NN(%r)" % self.elems
            

class PrEntry(object):
    def __init__(self, tree, dmin, d_query):
        """
        Constructor.

        arguments:
        d_query: distance d to searched query object
        """
        self.tree = tree
        self.dmin = dmin
        self.d_query = d_query

    def __lt__(self, other):
        return self.dmin < other.dmin

    def __repr__(self):
        return "PrEntry(tree:%r, dmin:%r)" % (self.tree, self.dmin)

    
class Entry(object):
    """
    
    The leafs and internal nodes of the M-tree contain a list of instances of
    this class.

    The distance to the parent is None if the node in which this entry is
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
        return "Entry(obj: %r, dist: %r, radius: %r, subtree: %r)" % (
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
                 mtree,
                 parent_node=None,
                 parent_entry=None,
                 entries=None):
        #There will be an empty node (entries set is empty) when the tree
        #is empty and there only is an empty root.
        #May also be empty during construction (create empty node then add
        #the entries later).
        self.mtree = mtree
        self.parent_node = parent_node
        self.parent_entry = parent_entry
        self.entries = set(entries) if entries else set()

    def __repr__(self): # pragma: no cover
        #entries might be big. Only prints the first few elements
        entries_str = '%s' % list(islice(self.entries, 2))
        if len(self.entries) > 2:
            entries_str = entries_str[:-1] + ', ...]'
            
        return "%s(parent_node: %s, parent_entry: %s, entries:%s)" % (
            self.__class__.__name__,
            self.parent_node.repr_class() \
                if self.parent_node else self.parent_node,
            self.parent_entry,
            entries_str
            
    )

    def repr_class(self): # pragma: no cover
        return self.__class__.__name__ + "()"

    def __len__(self): 
        return len(self.entries)

    @property
    def d(self):
        return self.mtree.d

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

    #TODO recomputes d(leaf, parent)!
    def set_entries_and_parent_entry(self, new_entries, new_parent_entry):
        self.entries = new_entries
        self.parent_entry = new_parent_entry
        self.parent_entry.radius = self.covering_radius_for(self.parent_entry.obj)
        self._update_entries_distance_to_parent()

    #wastes d computations if parent hasn't changed.
    #How to avoid? -> check if the new routing_object is the same as the old
    # (compare id(obj) not obj directly to prevent == assumption about object?)
    def _update_entries_distance_to_parent(self):
        if self.parent_entry:
            for entry in self.entries:
                entry.distance_to_parent = self.d(entry.obj,
                                                  self.parent_entry.obj)

    @abc.abstractmethod
    def add(self, obj): # pragma: no cover
        """Add obj into this subtree"""
        pass

    @abc.abstractmethod         
    def covering_radius_for(self, obj): # pragma: no cover
        """Compute the radius needed for obj to cover the entries of this node.
        """
        pass

    @abc.abstractmethod
    def search(self, query_obj, pr, nn, d_parent_query):
        pass
        

class LeafNode(AbstractNode):
    """A leaf of the M-tree"""
    def __init__(self,
                 mtree,
                 parent_node=None,
                 parent_entry=None,
                 entries=None):

        AbstractNode.__init__(self,
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

    def could_contain_results(self,
                              query_obj,
                              search_radius,
                              distance_to_parent, 
                              d_parent_query):
        """Determines without any d computation if there could be
        objects in the subtree belonging to the result.
        """
        if self.is_root():
            return True
        
        return abs(d_parent_query - distance_to_parent)\
                <= search_radius
        
    def search(self, query_obj, pr, nn, d_parent_query):
        for entry in self.entries:
            if self.could_contain_results(query_obj,
                                          nn.search_radius(),
                                          entry.distance_to_parent,
                                          d_parent_query):
                distance_entry_to_q = self.d(entry.obj, query_obj)
                if distance_entry_to_q <= nn.search_radius():
                    nn.update(entry.obj, distance_entry_to_q)
    
class InternalNode(AbstractNode):
    """An internal node of the M-tree"""

    def __init__(self,
                 mtree,
                 parent_node=None,
                 parent_entry=None,
                 entries=None):

        AbstractNode.__init__(self,
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
        for entry in self.entries:
            dist_to_obj[entry] = self.d(obj, entry.obj)

        def find_best_entry_requiring_no_covering_radius_increase():
            valid_entries = [e for e in self.entries
                             if dist_to_obj[e] <= e.radius]
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

    def set_entries_and_parent_entry(self, new_entries, new_parent_entry):
        AbstractNode.set_entries_and_parent_entry(self,
                                                  new_entries,
                                                  new_parent_entry)
        for entry in self.entries:
            entry.subtree.parent_node = self

    def could_contain_results(self,
                              query_obj,
                              search_radius,
                              entry,
                              d_parent_query):
        """Determines without any d computation if there could be
        objects in the subtree belonging to the result.
        """
        if self.is_root():
            return True
        
        parent_obj = self.parent_entry.obj
        return abs(d_parent_query - entry.distance_to_parent)\
                <= search_radius + entry.radius
            
    def search(self, query_obj, pr, nn, d_parent_query):
        for entry in self.entries:
            if self.could_contain_results(query_obj,
                                          nn.search_radius(),
                                          entry,
                                          d_parent_query):
                d_entry_query = self.d(entry.obj, query_obj)
                entry_dmin = max(d_entry_query - \
                                     entry.radius, 0)
                if entry_dmin <= nn.search_radius():
                    heappush(pr, PrEntry(entry.subtree, entry_dmin, d_entry_query))
                    entry_dmax = d_entry_query + entry.radius
                    if entry_dmax < nn.search_radius():
                        nn.update(None, entry_dmax)
                        
#A lot of the code is duplicated to do the same operation on the existing_node
#and the new node :(. Could prevent that by creating a set of two elements and
#perform on the (two) elements of that set.
#TODO: Ugly, complex code. Move some code in Node/Entry?
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
           value (the distance to existing_node.parent_entry).
    d: distance function.
    """
    assert existing_node.is_full()
    mtree = existing_node.mtree
    #type of the new node must be the same as existing_node
    #parent node, parent entry and entries are set later
    new_node = type(existing_node)(existing_node.mtree)
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
    assert entries1 and entries2, "Error during split operation. All the entries have been assigned to one routing_objects and none to the other! Should never happen since at least the routing objects are assigned to their corresponding set of entries"
    
    #must save the old entry of the existing node because it will have
    #to be removed from the parent node later
    old_existing_node_parent_entry = existing_node.parent_entry

    #Setting a new parent entry for a node updates the distance_to_parent in
    #the entries of that node, hence requiring d calls.
    #promote/partition probably did similar d computations.
    #How to avoid recomputations between promote, partition and this?
    #share cache (a dict) passed between functions?
    #memoization? (with LRU!).
    #    id to order then put the two objs in a tuple (or rather when fetching
    #      try both way
    #    add a function to add value without computing them
    #      (to add distance_to_parent)

    #TODO: build_entry in the node method?
    existing_node_entry = Entry(routing_object1,
                                None,#distance_to_parent set later
                                None,#covering_radius set later
                                existing_node)    
    existing_node.set_entries_and_parent_entry(entries1,
                                               existing_node_entry)

    new_node_entry = Entry(routing_object2, 
                           None,
                           None,
                           new_node)
    new_node.set_entries_and_parent_entry(entries2,
                                          new_node_entry)
                                          
    if existing_node.is_root():
        new_root_node = InternalNode(existing_node.mtree)

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
