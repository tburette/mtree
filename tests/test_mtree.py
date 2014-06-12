#python < 2.7 compatibility
try:
    import unittest2 as unittest
except ImportError:
    import unittest
import collections
from functools import reduce
#import both way because I want to access the public elements of the module
#without the mtree. prefix while also being able to access non public elements.
from mtree import *
import mtree


def whitebox(f):
    """Decorator specifying that the TestCase/test method is a whitebox test"""
    f.whitebox = True
    return f

def d_id(obj1, obj2):
    """dummy d implementation that works for any object.

    Respects all the condition for being a correct distance function.
    Works by comparing id."""
    return reduce(int.__sub__, sorted(map(id, (obj1, obj2)), reverse=True))

def d_int(i1, i2):
    """d implementation for integers"""
    return abs(i1 - i2)

class MTreeInitBadInput(unittest.TestCase):    
    def test_none_d(self):
        self.assertRaises(TypeError,MTree, None)

    def test_max_node_size(self):
        self.assertRaises(ValueError, MTree, d_id, 1)
        self.assertRaises(ValueError, MTree, d_id, 0)
        self.assertRaises(ValueError, MTree, d_id, -1)

class MTreeInitGoodInput(unittest.TestCase):
    def test_MTree_instantiation(self):
        def dummy1():pass
        def dummy2():pass
        tree = MTree(d_id,
                     16,
                     dummy1,
                     dummy2)
        self.assertEqual(d_id, tree.d)
        self.assertEqual(16, tree.max_node_size)
        self.assertEqual(dummy1, tree.promote)
        self.assertEqual(dummy2, tree.partition)
        self.assertEqual(0, len(tree))


def tree_len(tree):
    """Count the number of element in a tree by inspecting the nodes"""
    def node_len(node):
        if isinstance(node, mtree.InternalNode):
            return sum(node_len(e.subtree) for e in node.entries)
        else:
            return len(node.entries)
    return node_len(tree.root)

def node_objs(node):
    """Return objects in the node and subnodes"""
    if isinstance(node, mtree.InternalNode):
        result = []
        for n in node.entries:
            result.extend(node_objs(n.subtree))
        return result
    else:
        return list(map(lambda e: e.obj, node.entries))

def tree_objs(tree):
    """Returns all the objects of the tree"""
    return node_objs(tree.root)


class MTreeLength(unittest.TestCase):
    def test_len(self):
        tree = MTree(d_id)
        self.assertEqual(len(tree), 0)
        tree.add(1)
        self.assertEqual(len(tree), 1)
        tree.add(10)
        self.assertEqual(len(tree), 2)
        tree.add_all([1, 2, 3])
        self.assertEqual(len(tree), 5)

@whitebox
class MTreeAdd(unittest.TestCase):
    def test_add(self):
        tree = MTree(d_id)
        self.assertEqual(tree_len(tree), 0)
        self.assertEqual(set(tree_objs(tree)), set())
        tree.add(1)
        self.assertEqual(tree_len(tree), 1)
        self.assertEqual(set(tree_objs(tree)), set([1]))
        tree.add(10)
        self.assertEqual(tree_len(tree), 2)
        self.assertEqual(set(tree_objs(tree)), set([1, 10]))
        tree.add_all([1, 2, 3])
        self.assertEqual(tree_len(tree), 5)
        self.assertEqual(sorted(tree_objs(tree)), [1, 1, 2, 3, 10])

    def test_add2(self):
        tree = MTree(lambda i1, i2: abs(i1 - i2))
        tree.add_all(range(34))

    def test_add_with_split(self):
        max_sizes = [2, 3, 4, 99]
        for max_size in max_sizes:
            tree = MTree(d_int, max_size)
            objs = range(100)
            for o in objs:
                tree.add(o)
            self.assertEqual(len(tree), 100)
            self.assertEqual(sorted(tree_objs(tree)), list(range(100)))
        
    def test_add_all(self):
        tree = MTree(d_id)
        tree.add_all([1])
        self.assertEqual(sorted(tree_objs(tree)), [1])
        tree.add_all([2, 3])
        self.assertEqual(sorted(tree_objs(tree)), [1, 2, 3])
        tree.add_all([2, 3, 4])
        self.assertEqual(sorted(tree_objs(tree)), [1, 2, 2, 3, 3, 4])
        
@whitebox
class TestEntry(unittest.TestCase):
    def test_Entry_creation(self):
        e = mtree.Entry(1, 2, 3, 4)
        self.assertEqual(e.obj, 1)
        self.assertEqual(e.distance_to_parent, 2)
        self.assertEqual(e.radius, 3)
        self.assertEqual(e.subtree, 4)

@whitebox
class TestAbstractNode(unittest.TestCase):
    """Tests for AbstractNode

    I know that LeafNode and InternalNode share common behaviour trough
    AbstractNode. I cheat by testing abstractNode methods through Leaf only."""
    def test_init(self):
        tree = MTree(d_int)
        parent_node = mtree.InternalNode(d_int, tree)
        parent_entry = mtree.Entry(1)
        leaf = mtree.LeafNode(d_int, tree, parent_node, parent_entry, None)
        self.assertEqual(leaf.d, d_int)
        self.assertEqual(leaf.mtree, tree)
        self.assertEqual(leaf.parent_node, parent_node)
        self.assertEqual(leaf.parent_entry, parent_entry)
        self.assertEqual(leaf.entries, set())
    def test_len(self):
        self.assertEqual(len(make_leaf_node(d_int, [1, 3, 5])), 3)
        self.assertEqual(len(make_leaf_node(d_int, [1, ])), 1)
        self.assertEqual(len(make_leaf_node(d_int, [])), 0)

    def test_is_full(self):
        self.assertTrue(make_leaf_node(d_int, [1, 3, 5], 3).is_full())
        self.assertTrue(make_leaf_node(d_int, [1, 2], 2).is_full())
        self.assertFalse(make_leaf_node(d_int, [1, 2], 3).is_full())
        self.assertFalse(make_leaf_node(d_int, [], 3).is_full())

    def test_is_empty(self):
        self.assertFalse(make_leaf_node(d_int, [1, 3, 5], 3).is_empty())
        self.assertFalse(make_leaf_node(d_int, [1, 2], 2).is_empty())
        self.assertFalse(make_leaf_node(d_int, [1, 2], 3).is_empty())
        self.assertTrue(make_leaf_node(d_int, [], 3).is_empty())

    def test_is_root(self):
        self.assertTrue(make_leaf_node(d_int, [1, 3, 5], 3).is_root())
        leaf = make_leaf_and_internal_node(d_int, 1, [1, 2])
        root = leaf.mtree.root
        self.assertFalse(leaf.is_root())
        self.assertTrue(root.is_root())

    def test_remove_entry(self):
        leaf = make_leaf_node(d_int, [1, 3, 5, 7])
        entries = set(leaf.entries)
        while entries:
            e = entries.pop()
            leaf.remove_entry(e)
            self.assertEqual(leaf.entries, entries)
        self.assertEqual(leaf.entries, set())

    def test_remove_entry_error(self):
        leaf = make_leaf_node(d_int, [1, 3, 5, 7])
        self.assertRaises(KeyError, leaf.remove_entry, 2)

    def test_add_entry(self):
        leaf = make_leaf_node(d_int, [1], 3)
        
        leaf.add_entry(mtree.Entry(2))
        self.assertEqual(entries_to_objs(leaf.entries), set([1, 2]))
        leaf.add_entry(mtree.Entry(3))
        self.assertEqual(entries_to_objs(leaf.entries), set([1, 2, 3]))
        self.assertRaises(ValueError, leaf.add_entry, mtree.Entry(4))

@whitebox
class TestLeafNode(unittest.TestCase):
    def test_covering_radius_for(self):
        leaf = make_leaf_node(d_int, [1, 3, 5, 7])
        self.assertEqual(leaf.covering_radius_for(5), 4)
        self.assertEqual(leaf.covering_radius_for(7), 6)
        self.assertEqual(leaf.covering_radius_for(10), 9)
        self.assertEqual(
            make_leaf_node(d_int, [1]).covering_radius_for(1),
            0)

    def test_add(self):
        """Test adding an object to a leaf but doesn't test splits"""
        leaf = make_leaf_and_internal_node(d_int, 2, leaf_objs=[1, 3])
        leaf.add(5)
        self.assertEqual(len(leaf), 3)
        e = entry_of_obj(leaf.entries, 5)
        self.assertNotEqual(e, None)
        self.assertEqual(e.obj, 5)
        self.assertEqual(e.distance_to_parent, 3)
        self.assertEqual(e.radius, None)
        self.assertEqual(e.subtree, None)
        
@whitebox       
class TestInternalNode(unittest.TestCase):
    def test_covering_radius_for(self):
        tree = MTree(d_int)
        internal = mtree.InternalNode(d_int, tree)
        internal.add_entry(mtree.Entry(1, None, 1))
        internal.add_entry(mtree.Entry(8, None, 2))
        self.assertEqual(internal.covering_radius_for(0), 10)
        self.assertEqual(internal.covering_radius_for(10), 10)
        self.assertEqual(internal.covering_radius_for(1), 9)
        self.assertEqual(internal.covering_radius_for(11), 11)

    def test_add(self):
        routing_obj1 = 10
        routing_obj2 = 20
        internal = make_two_leaves_and_one_internal(d_int,
                                                    routing_obj1,
                                                    [],
                                                    routing_obj2,
                                                    [])
        entry1 = next((e for e in internal.entries \
                     if e.obj == routing_obj1))
        leaf1 = entry1.subtree
        entry2 = next((e for e in internal.entries \
                     if e.obj == routing_obj2))
        leaf2 = entry2.subtree
        
        internal.add(9)
        #internal node isn't borked
        self.assertEqual(len(internal), 2)
        self.assertEqual(internal.parent_node, None)
        self.assertEqual(internal.parent_entry, None)
        #added to the right leaf
        self.assertEqual(len(leaf1), 1)
        self.assertEqual(len(leaf2), 0)
        #radius increased correctly
        self.assertEqual(entry1.radius, 1)
        self.assertEqual(entry2.radius, 0)
        #entries aren't borked
        self.assertEqual(entry1.obj, 10)
        self.assertEqual(entry1.distance_to_parent, None)
        self.assertEqual(entry1.subtree, leaf1)
        self.assertEqual(entry2.obj, 20)
        self.assertEqual(entry2.distance_to_parent, None)
        self.assertEqual(entry2.subtree, leaf2)

        internal.add(11)
        self.assertEqual(len(leaf1), 2)
        self.assertEqual(len(leaf2), 0)
        #radius didn't change
        self.assertEqual(entry1.radius, 1)
        self.assertEqual(entry2.radius, 0)

@whitebox
class TestSplit(unittest.TestCase):
    def test_split(self):
        leaf = make_leaf_node(d_int, [1, 3, 5, 7], 4)
        tree = leaf.mtree
        
        mtree.split(leaf, mtree.Entry(9), d_int)
        all_objs = [1, 3, 5, 7, 9]
        
        internal = tree.root
        self.assertTrue(isinstance(internal, mtree.InternalNode))
        self.assertTrue(internal.parent_node is None)
        self.assertTrue(internal.parent_entry is None)
        self.assertEqual(len(internal.entries), 2)
        for e in internal.entries:
            self.assertTrue(e.obj in all_objs)
            self.assertTrue(e.distance_to_parent is None)
            self.assertFalse(e.subtree is None)
        def leafs(entries):
            g = (e for e in entries)
            return (next(g).subtree, next(g).subtree)
        leaf1, leaf2 = leafs(internal.entries)
        self.assertTrue(leaf == leaf1 or leaf == leaf2)
        self.assertTrue(isinstance(leaf1, mtree.LeafNode))
        self.assertTrue(isinstance(leaf2, mtree.LeafNode))
        self.assertEqual(leaf1.parent_node, internal)
        self.assertEqual(leaf1.parent_node, internal)
        self.assertTrue(leaf1.parent_entry in internal.entries)
        self.assertTrue(leaf2.parent_entry in internal.entries)
        self.assertEqual(len(leaf1)+len(leaf2), 5)
        self.assertEqual(sorted(node_objs(leaf1) + node_objs(leaf2)),
                         all_objs)
        
        
        
        
def entries_to_objs(entries):
    return set(map(lambda e: e.obj, entries))

def entry_of_obj(entries, obj):
    """Returns the (first) entry withthe specified obj """
    for e in entries:
        if e.obj == obj:
            return e
    return None

        
def make_leaf_node(d, objs=[], max_size=None):
    if max_size:
        tree = MTree(d, max_size)
    else:
        tree = MTree(d)

    if tree.max_node_size < len(objs):
        raise ValueError('More objects than the maximum size of a node')
    leaf = tree.root
    leaf.entries = set(map(mtree.Entry, objs))
    tree.size = len(objs)
    return leaf


def make_leaf_and_internal_node(d, routing_object, leaf_objs=[]):
    """Creates an mtree with one leaf node which has one parent.

    This is not a tree which would happen in reality: there should be at least
    two nodes connected to a parent.
    """
    tree = MTree(d)
    tree.size = len(leaf_objs)
    leaf = tree.root
    internal = mtree.InternalNode(d, tree)
    tree.root = internal
    leaf.parent_node = internal
    #entry of leaf in the root
    leaf_entry = mtree.Entry(routing_object,
                             None,
                             leaf.covering_radius_for(routing_object),
                             leaf)
    internal.add_entry(leaf_entry)
    leaf.parent_entry = leaf_entry
    leaf.entries = set(map(lambda o: mtree.Entry(o, d(o, routing_object)),
                           leaf_objs))
    return leaf


def make_two_leaves_and_one_internal(d,
                                     routing1,
                                     leaf_objs1,
                                     routing2,
                                     leaf_objs2):
    tree = make_leaf_and_internal_node(d, routing1, leaf_objs1).mtree
    internal = tree.root
    entry2 = mtree.Entry(routing2)
    leaf2 = mtree.LeafNode(d, tree, internal, entry2)
    leaf2.entries = set(map(lambda o: mtree.Entry(o, d(o, routing2)),
                           leaf_objs2))
    entry2.subtree = leaf2
    entry2.radius = leaf2.covering_radius_for(routing2)
    internal.add_entry(entry2)
    return internal

@whitebox
class TestM_LB_DIST_leaf_and_internal(unittest.TestCase):
    #Leaf whose entries are examined has a parent
    data_confirmed = \
        [(make_leaf_and_internal_node(d_int, 5, [1, 3, 5, 7, 20]), [5, 20]),
         (make_leaf_and_internal_node(d_int, 1, [1, 3, 7]), [1, 7]),
         (make_leaf_and_internal_node(d_int, 0, [1, 3, 7]), [0, 7]),
         (make_leaf_and_internal_node(d_int, 1, [1, 3]), [1, 3]),
         (make_leaf_and_internal_node(d_int, 3, [1, 3]), [1, 3]),
         (make_leaf_and_internal_node(d_int, 3, [1, 3, 5]), [1, 3]),
         ]

    #Leaf whose entries are examined has a parent
    data_non_confirmed = \
        [(make_leaf_and_internal_node(d_int, 5, [1, 3, 5, 7, 20]), [1, 20]),
         (make_leaf_and_internal_node(d_int, 1, [1, 3, 7]), [1, 7]),
         (make_leaf_and_internal_node(d_int, 0, [1, 3, 7]), [1, 7]),
         (make_leaf_and_internal_node(d_int, 1, [1, 3]), [1, 3]),
         (make_leaf_and_internal_node(d_int, 3, [1, 3]), [1, 3]),
         (make_leaf_and_internal_node(d_int, 3, [1, 3, 5]), [1, 5]),
         ]

    #Leaf with no parent (leaf is root)
    data_leaf = [(make_leaf_node(d_int, [1, 3]), [1, 3]),
            (make_leaf_node(d_int, [1, 3, 5, 7], 5), [1, 7]),
            (make_leaf_node(d_int, [1, 3, 5, 7], 4), [1, 7])]
        
    def execute_test(self, promote, data):
        for (node, expected_result) in data:
            result = promote(node.entries,
                             node.parent_entry,
                             d_int)
            #promote returns entries but we want the object they contain.
            #The two entry returned may be in any order so we also
            #sort them to compare easily with the expected result
            result = sorted(result)
            self.assertEqual(result, expected_result)

    def test_M_LB_DIST_confirmed(self):
        self.execute_test(M_LB_DIST_confirmed, self.data_confirmed)

    def test_M_LB_DIST_non_confirmed(self):
        self.execute_test(M_LB_DIST_non_confirmed, self.data_non_confirmed)

    def test_M_LB_DIST_non_confirmed_on_leaf(self):
        self.execute_test(M_LB_DIST_non_confirmed, self.data_leaf)

    #Actually delegates its work to the non confirmed version
    def test_M_LB_DIST_confirmed_on_leaf(self):
        self.execute_test(M_LB_DIST_confirmed, self.data_leaf)


PartitionData = collections.namedtuple('PartitionData',
                                         'node obj1 assigned1 obj2 assigned2')
@whitebox
class TestGeneralizedHyperplane(unittest.TestCase):
    data = [PartitionData(node=make_leaf_node(d_int, [1, 3, 5, 7]),
                          obj1=1, assigned1=set([1, 3]),
                          obj2=7, assigned2=set([5, 7])),
            #all in one set
            PartitionData(node=make_leaf_node(d_int, [1, 2, 3, 4, 5], 5),
                          obj1=1, assigned1=set([1, 2, 3, 4, 5]),
                          obj2=50, assigned2=set([])),
            PartitionData(node=make_leaf_node(d_int, [1, 2, 3, 4, 5], 5),
                          obj1=-10, assigned1=set([]),
                          obj2=1, assigned2=set([1, 2, 3, 4, 5])),
            #routing objects are not in the entries
            PartitionData(node=make_leaf_node(d_int, [1, 2, 3, 4, 5, 6], 6),
                          obj1=0, assigned1=set([1, 2, 3]),
                          obj2=7, assigned2=set([4, 5, 6])),
            #empty
            PartitionData(node=make_leaf_node(d_int, []),
                          obj1=1, assigned1=set([]),
                          obj2=10, assigned2=set([])),
            #one elem
            PartitionData(node=make_leaf_node(d_int, [1]),
                          obj1=2, assigned1=set([1]),
                          obj2=10, assigned2=set([])),
            #two elems
            PartitionData(node=make_leaf_node(d_int, [1, 2]),
                          obj1=1, assigned1=set([1]),
                          obj2=2, assigned2=set([2]))]

    def test_generalized_hyperplane(self):
        for test in self.data:
            result = generalized_hyperplane(test.node.entries,
                                            test.obj1,
                                            test.obj2,
                                            d_int)
            def entry_set_to_obj_set(entry_set):
                return set(map(lambda e: e.obj, entry_set))                
            result = tuple(map(lambda entries: entry_set_to_obj_set(entries),
                               result))

            self.assertEqual(result,
                             (test.assigned1, test.assigned2))
            

if __name__ == '__main__':
    unittest.main()
    
