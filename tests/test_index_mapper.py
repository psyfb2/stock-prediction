import unittest

import pandas as pd
import numpy as np

from data_preprocessing.index_mapper import IndexMapper

class TestDataCollection(unittest.TestCase):
    def test_index_mapper(self):
        window_size = 3
        lengths = np.array([5, 4, 6])

        correct_mapping = {
            0: 2,
            1: 3,
            2: 4,

            3: 7,
            4: 8,

            5: 11,
            6: 12,
            7: 13,
            8: 14
        }

        idx_mapper = IndexMapper(lengths, window_size)
        for k, v in correct_mapping.items():
            self.assertEqual(idx_mapper(k), v)

        self.assertEqual(len(idx_mapper), len(correct_mapping))

    def test_index_mapper_monkey_test(self):
        for _ in range(10):
            window_size = np.random.randint(1, 10)
            lengths = np.random.randint(window_size, window_size * 10, size=np.random.randint(1, 10))

            datasets = [np.zeros((length, 5)) for length in lengths]

            correct_mapping = {}
            idx = 0
            true_idx = 0
            for i in range(len(datasets)):
                for j in range(datasets[i].shape[0]):
                    if j >= window_size - 1:
                        correct_mapping[idx] = true_idx
                        idx += 1
                    true_idx += 1

            idx_mapper = IndexMapper(lengths, window_size)  # uses way less space than the method above (O(1) vs O(n))
            for k, v in correct_mapping.items():
                self.assertEqual(idx_mapper(k), v, f"using window_size={window_size}, lengths={lengths}, idx={k}\n"
                                                   f"correct_mapping={correct_mapping}")
            
            self.assertEqual(len(idx_mapper), len(correct_mapping))
        
if __name__ == "__main__":
    unittest.main()
