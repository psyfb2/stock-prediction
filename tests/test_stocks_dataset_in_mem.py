import unittest

import numpy as np

from data_preprocessing.dataset import StocksDatasetInMem


class TestDataCollection(unittest.TestCase):
    def test_window_mat1(self):
        window_size = 4

        mat = np.array(
            [
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
                [19, 20, 21],
                [22, 23, 24],
                [25, 26, 27],
                [28, 29, 30]
            ]
        )

        windowed_mat = np.array(
            [
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
                [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            ]
        )

        self.assertTrue((StocksDatasetInMem.window_matrix(mat, window_size) == windowed_mat).all())
    
    def test_window_mat2(self):
        window_size = 4

        mat = np.array(
            [
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
                [19, 20, 21],
            ]
        )

        windowed_mat = np.array(
            [
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            ]
        )

        self.assertTrue((StocksDatasetInMem.window_matrix(mat, window_size) == windowed_mat).all())
    
    def test_get_full_data_matrix(self):
        window_size = 4

        features = np.array(
            [
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
                [19, 20, 21],
                [22, 23, 24],
                [25, 26, 27],
                [28, 29, 30],

                [31, 32, 33],
                [34, 35, 36],
                [37, 38, 39],
                [40, 41, 42],
                [43, 44, 45],

                [46, 47, 48],
                [49, 50, 51],
                [52, 53, 54],
                [55, 56, 57]
            ]
        )
        lengths = [7, 5, 4]
        labels = np.array(
            [
                0, 0, 0, 1, 1, 0, 1,  
                0, 1, 0, 1, 0, 
                1, 0, 0, 1
            ]
        )

        expected_X = np.array(
            [
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
                [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],

                [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
                [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45],

                [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
            ]
        )
        expected_y = np.array(
            [
                1, 1, 0, 1,
                1, 0,
                1
            ]
        )

        dataset = StocksDatasetInMem(
            tickers=None, features_to_use=None,
            vix_features_to_use=None,
            start_date=None, end_date=None, label_config=None, 
            num_candles_to_stack=window_size,
            candle_size=None, features=features, labels=labels, 
            lengths=lengths
        )

        X, y = dataset.get_full_data_matrix()

        self.assertTrue((X == expected_X).all())
        self.assertTrue((y == expected_y).all())

        
if __name__ == "__main__":
    unittest.main()
