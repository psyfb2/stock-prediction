import unittest

import pandas as pd
import numpy as np

from data_preprocessing.labelling import binary_label_tp_tsl

class TestDataCollection(unittest.TestCase):
    def test_binary_label_tp_tsl(self):
        df = pd.DataFrame(
            {
                "o": [95,  100, 100, 102, 102, 100, 100, 100, 100],
                "c": [95,  100, 100, 102, 102, 100, 100, 100, 100],
                "h": [110, 102, 101, 105, 105, 100, 100, 100, 100],
                "l": [90,  99,  96,  96,  96,  100, 100, 90,  100],
                # expected result
                "r": [0,   1,   0,   0,   0,   0,   0, np.nan, np.nan]
            }
        )

        labels = binary_label_tp_tsl(df, 0.05, 0.05)
        self.assertTrue(df["r"].equals(labels))
    
    def test_binary_label_tp_tsl2(self):
        df = pd.DataFrame(
            {
                "o": [20, 19, 20, 21, 20, 19],
                "c": [20, 19, 20, 21, 20, 19],
                "h": [21, 21, 22, 21, 21, 20],
                "l": [19, 19, 20, 20, 19, 17.9],
                # expected result
                "r": [0,  0,  0,  0,  0, np.nan]
            }
        )

        labels = binary_label_tp_tsl(df, 0.2, 0.1)
        self.assertTrue(df["r"].equals(labels))
    
    def test_binary_label_tp_tsl3(self):
        df = pd.DataFrame(
            {
                "o": [20, 22, 26, 28, 20, 30],
                "c": [20, 22, 26, 28, 20, 30],
                "h": [21, 22, 26, 28, 20, 30],
                "l": [19, 22, 26, 28, 20, 30],
                # expected result
                "r": [1,  0,  0,  1, np.nan, np.nan]
            }
        )

        labels = binary_label_tp_tsl(df, 0.1, 0.2)
        self.assertTrue(df["r"].equals(labels))
    

    def test_binary_label_tp_tsl_all_nan(self):
        df = pd.DataFrame(
            {
                "o": [50, 51, 50, 49, 51, 52],
                "c": [51, 50, 51, 50, 51, 52],
                "h": [51, 52, 52, 50, 52, 50],
                "l": [50, 49, 49, 49, 50, 49],
                # expected result
                "r": [np.nan] * 6
            }
        )

        labels = binary_label_tp_tsl(df, 0.15, 0.1)
        self.assertTrue(df["r"].equals(labels))
       

if __name__ == "__main__":
    unittest.main()
