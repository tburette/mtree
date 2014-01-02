#TODO: add: appliquer optimisation qui utilise d du parent pour reduire d faits
#infinity? (cf k-NN search)
#implement only one promote and partition policies but code so that
#other can be added later
#split: handle case where a node had no parent but now has one.
# Must compute and assign d(Oj, Op)

#from abc import ABCMeta, abstractmethod, abstractproperty

__all__ = ['MTree']

#TODO: node size : 32 is arbitrary. Define a reasonable default value
class MTree():
    def __init__(self, d, maxNodeSize=32):
        """
        Creates a new MTree.

        d: metric of the metric space.
        maxNodeSize: optional. Maximum number of entries in a node of
        the M-tree
        """
        self.size = 0
        self.root = Leaf(d, maxNodeSize)
        self.d = d

    def __len__(self):
        return self.size

    def add(self, obj):
        """
        Adds an object into the M-tree
        """
        self.root.add(obj)
        self.size += 1

#TODO: remove Node and use duck typing instead? No one will need to perform
# inspection on these class anyway as they are private
'''
class Node():
    __metaclass__ = ABCMeta
    """Abstract class. Base class of Leaf and internal node"""
    def __init__(self, d, maxNodeSize, parent=None):
        self.d = d
        self.maxNodeSize = maxNodeSize
        self.parent = parent

    @abstractmethod
    def isinternal(self):
        abstract
'''

class LeafEntry():
    """
    Entry found in the leafs of the M-tree

    The distance to the parent is None if the leaf in which this entry is
    stored has no parent"""
    def __init__(self, obj, distance_to_parent=None):
        self.obj = obj
        self.distance_to_parent = distance_to_parent

class InternalNodeEntry():
    """
    Entry found in the internal nodes of the M-tree

    The distance to the parent is None if the leaf in which this entry is
    stored has no parent
    """
    def __init__(self,
                 obj,
                 covering_radius,
                 distance_to_parent=None,
#TODO: shouldn't an internal node always have a covering tree?
                 covering_tree=None):
        self.obj = obj
        self.covering_tree = covering_tree
        self.covering_radius = covering_radius
        self.distance_to_parent = distance_to_parent
        
    
class Leaf(Node):
    """A leaf of the M-tree"""
    def __init__(self, d, maxNodeSize, parent = None, entries=set()):
        self.d = d
        self.maxNodeSize = maxNodeSize
        self.parent = parent
        self.entries = entries

    def hasparent(self):
        return bool(self.parent)
    
    def isinternal(self):
        return false

    def add(self, obj):
        raise Exception('Not yet implemented')
        if(len(entries) < maxNodeSize):
            pass
            

    
class InternalNode(Node):
    """An internal node of the M-tree"""
    def __init__(self, d, maxNodeSize, parent = None, entries=set()):
        self.d = dg
        self.maxNodeSize = maxNodeSize
        self.parent = parent
        self.entries = entries

    def hasparent(self):
        return bool(self.parent)
    
    def isinternal(self):
        return false
