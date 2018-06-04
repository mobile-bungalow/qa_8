import utils
import unittest

class ListSorting(unittest.TestCase):
    def test_most_freq(self):
        self.assertEqual(utils.check_for_group([('But', 'NNP'), ('Martin', 'NNP'),
         (',', ','), ('Edgar', 'NNP'), (',', ','), ('and', 'CC'), ('Cindy', 'NNP'),
          ('did', 'VBD'), ("n't", 'RB'), ('like', 'VB'), ('Thomas', 'NNP'), ('.', '.')]),(6,'Martin , Edgar , and Cindy'))
          
if __name__ == '__main__':
    unittest.main()