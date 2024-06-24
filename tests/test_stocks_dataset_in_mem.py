import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

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
    
    @patch.object(StocksDatasetInMem, "load_dataset_in_mem")
    def test_get_full_data_matrix(self, mock_load_dataset_in_mem):
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

        data_df = pd.DataFrame(
            {
                "f1": features[:, 0],
                "f2": features[:, 1],
                "f3": features[:, 2],
                "labels": labels
            }
        )
        for c in list("tochlv"):
            data_df[c] = 0

        mock_load_dataset_in_mem.return_value = (
            data_df, lengths, None, None
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
            candle_size=None, sectors=None
        )

        X, y = dataset.get_full_data_matrix()

        self.assertTrue((X == expected_X).all())
        self.assertTrue((y == expected_y).all())
    
    @patch.object(StocksDatasetInMem, "load_dataset_in_mem")
    def test_saving_and_loading(self, mock_load_dataset_in_mem):
        window_size = 2

        features = np.array(
            [
                [10, 11, 12, 1],
                [13, 14, 15, 2],
                [16, 17, 18, 3],
                [19, 20, 21, 4],
                [22, 23, 24, 3],
                [25, 26, 27, 2],
                [28, 29, 30, 1],

                [31, 32, 33, 5],
                [34, 35, 36, 1],
                [37, 38, 39, 2],
                [40, 41, 42, 3],
                [43, 44, 45, 2],

                [46, 47, 48, 1],
                [49, 50, 51, 2],
                [52, 53, 54, 3],
                [55, 56, 57, 4]
            ]
        )
        lengths = [7, 5, 4]
        labels = np.array(
            [
                1, 1, 1, 0, 0, 0, 1,  
                0, 1, 1, 1, 0, 
                1, 1, 1, 1
            ]
        )

        data_df = pd.DataFrame(
            {
                "f1": features[:, 0],
                "f2": features[:, 1],
                "f3": features[:, 2],
                "c":  features[:, 3],
                "labels": labels
            }
        )
        for c in list("tohlv"):
            data_df[c] = 0

        mock_load_dataset_in_mem.return_value = (
            data_df, lengths, None, None
        )

        dataset = StocksDatasetInMem(
            tickers=None, features_to_use=None,
            vix_features_to_use=None,
            start_date=None, end_date=None, label_config=None, 
            num_candles_to_stack=window_size,
            candle_size=None, sectors=None, save_path="tests/"
        )

        dataset = StocksDatasetInMem(
            tickers=None, sectors=None, features_to_use=None, vix_features_to_use=None,
            start_date=None, end_date=None, label_config={
                "label_function": "binary_label_close_higher",
                "kwargs": {
                    "candles_ahead": 2
                }
            },
            num_candles_to_stack=window_size,
            stds=None, means=None, load_path="tests/", recalculate_labels=True
        )

        self.assertTrue(
            np.all(
                np.isclose(
                    dataset.features, 
                    np.array(
                        [
                            [10, 11, 12],
                            [13, 14, 15],
                            [16, 17, 18],
                            [19, 20, 21],
                            [22, 23, 24],

                            [31, 32, 33],
                            [34, 35, 36],
                            [37, 38, 39],

                            [46, 47, 48],
                            [49, 50, 51],
                        ]
                    )
                )
            )
        )

        self.assertTrue(np.all(dataset.lengths == [5, 3, 2]))
        self.assertTrue(np.all(dataset.labels == [1, 1, 0, 0, 0, 0, 1, 0, 1, 1]))

        os.remove("tests/" + dataset.DATASET_FILENAME)
        os.remove("tests/" + dataset.LENGTHS_FILENAME)

        
if __name__ == "__main__":
    unittest.main()
