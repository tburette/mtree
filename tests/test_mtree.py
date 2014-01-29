import unittest
import mtree

from mtree import *

#dummy d implementation that respects all the conditions 
def d(obj1, obj2):
    return reduce(int.__sub__, sorted(map(id, (obj1, obj2)), reverse=True))

class MTreeInitBadInput(unittest.TestCase):    
    def test_none_d(self):
        self.assertRaises(TypeError,MTree, None)

    def test_max_node_size(self):
        self.assertRaises(ValueError, MTree, d, 1)
        self.assertRaises(ValueError, MTree, d, 0)
        self.assertRaises(ValueError, MTree, d, -1)

class MTreeGoodInput(unittest.TestCase):
    def test_MTreeinstantiation(self):
        def dummy1():pass
        def dummy2():pass
        tree = MTree(d,
                     16,
                     dummy1,
                     dummy2)
        self.assertEqual(d, tree.d)
        self.assertEqual(16, tree.max_node_size)
        self.assertEqual(dummy1, tree.promote)
        self.assertEqual(dummy2, tree.partition)
        self.assertEqual(0, len(tree))
                     



if __name__ == '__main__':
    unittest.main()
    
