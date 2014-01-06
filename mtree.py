#TODO: add: appliquer optimisation qui utilise d du parent pour reduire d faits
#infinity? (cf k-NN search)
#implement only one promote and partition policies but code so that
# other can be added later
#split: handle case where a node had no parent but now has one.
# Must compute and assign d(Oj, Op)
#TODO: specify in docstring exactly what can return d :
# (something that work like) a number
#TODO: specify in docstring what exactly an obj can be (anything..)

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
    The leafs of the M-tree contain a list of instances of this class

    The distance to the parent is None if the leaf in which this entry is
    stored has no parent"""
    def __init__(self, obj, distance_to_parent=None):
        self.obj = obj
        self.distance_to_parent = distance_to_parent

class InternalNodeEntry():
    """
    The internal nodes of the M-tree contain a list of instances of this class

    The distance to the parent is None if the leaf in which this entry is
    stored has no parent
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

    def isinternal(self):
        return false

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
        
    def isinternal(self):
        return true

    def add(self, obj):
        #V1
        #require to define eq and hash in InternalNodeEntry
        #Why not use the default hash?
        distance = {}
        for e in self.entries:
            distance[e] = d(obj, e)
            
        valid_entries = filter(lambda e : distance[e] <= e.radius,
                              self.entries)
        
        if valid_entries:
            best_entry = min(distance, key=distance.get)
            best_entry.add(obj)

        """
        For posterity:
        
        #a named tuple which contain a (leaf or internalNode) entry and
        #an associated distance.
        #The distance should be between the object in the entry and
        #another object
        from collections import namedtuple
        DistEntry = namedtuple('DistEntry', 'distance entry')
        def distEntrylt(self, other):
            return self.distance < other.distance
        DistEntry.__lt__ = distEntrylt
        del distEntrylt

        #V2
        distEntries = map(lambda e :
                              DistEntry(distance=d(obj, e.obj), entry=e),
                          self.entries)
        
        validDistEntries = filter(lambda e : e.distance <= e.entry.radius,
                                  distEntries)
        
        if validDistEntries:            
            best_entry = min(validDistEntries).entry
            best_entry.add(obj)
        
        #V3
        best_entry = None
        best_entry_distance = None
#TODO: add: appliquer optimisation qui utilise d du parent pour reduire d faits
        for entry in self.entries:
            distance = self.d(obj, entry.obj)
            if distance < entry.radius:
                if not(best_entry) or distance < best_entry_distance:
                    best_entry = entry
                    best_entry_distance = distance
        best_entry.add(obj)

        #V4
        a proxy that only intersept the computed distance and defer
        for the rest

        #V5
        modify entries to temporarily contain a new distance field
        field is removed after
        """
