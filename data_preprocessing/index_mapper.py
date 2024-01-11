from typing import List

import numpy as np

class IndexMapper:
    def __init__(self, lengths: List[int], window_size: int):
        """ Init IndexMapper. This is useful for an (N, D) dataset
        to which windowing of window_size must be applied, but the (N, D)
        dataset is made up of different sections, each with length specified
        in lengths and the windowing should not cross any of these boundaries specified
        by the lengths.

        Args:
            lengths (List[int]): length of each section in an (N, D) dataset in-order.
            window_size (int): window size which will be used
        """
        self.lengths = lengths
        self.window_size = window_size
        self.N = sum(lengths)

        indicies = [0]
        cur_sum = 0
        for i in range(len(lengths)):
            cur_sum += lengths[i]
            indicies.append(cur_sum - (i + 1) * (window_size - 1))
        self.indicies = np.array(indicies)
    
    def __call__(self, idx: int) -> int:
        """ Get index within (N, D) dataset
        where idx represents index excluding all
        elements in (N, D) data which are within window_size
        (i.e. lengths specifies the portions of (N, D) dataset,
        windowing should never cross the boundaries specified by lengths).

        Args:
            idx (int): index within (N, D) dataset excluding all elements
                within window_size of a boundary specified by lengths.
        Returns
            int: true index between 0 and N - 1 == sum(lengths) - 1
        """
        # Time: O(log(len(lengths)))
        k = np.searchsorted(self.indicies, idx, side="right")
        return k * (self.window_size - 1) + idx
    
    def __len__(self) -> int:
        return self.N - (len(self.lengths) * (self.window_size - 1))
