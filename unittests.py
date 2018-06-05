import utils
import unittest

def consume_loop():
	ret = []
	l = [1,2,3,4,5,6,7,8,9,10].__iter__()
	for i,n in enumerate(l):
		ret += [n]
		if n == 3:
			utils.consume(l,4)

	return ret

class ListSorting(unittest.TestCase):
    
    def test_most_freq(self):
    	#group chunking function
        self.assertEqual(utils.check_for_group([('But', 'NNP'), ('Martin', 'NNP'),
         (',', ','), ('Edgar', 'NNP'), (',', ','), ('and', 'CC'), ('Cindy', 'NNP'),
          ('did', 'VBD'), ("n't", 'RB'), ('like', 'VB'), ('Thomas', 'NNP'), ('.', '.')]),(6,'Martin , Edgar , and Cindy'))
    
    def test_consume(self):
    	#consumption of iterators
    	self.assertEqual(consume_loop(),[1,2,3,8,9,10])

    def test_similarity(self):
    	#testing similarity
    	


if __name__ == '__main__':
    unittest.main()