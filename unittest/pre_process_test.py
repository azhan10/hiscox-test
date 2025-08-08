import unittest
import sys, pathlib

from pathlib import Path

sys.path.append(str(pathlib.Path().resolve().parent) + "/scripts/")

import pre_process as preprocess_module


"""

I could add more tests like:
    - Testing the range a column has x number of unique values
    - Testing the if the min max scaler columns have changed

"""


class TestPreProcessData(unittest.TestCase):

    
    """
    
    Testing if the data shape is correct
    
    """
    def test_data_shape_path(self):
        pre_process_class = preprocess_module.PreProcessData(query="")
        data_original_df = pre_process_class.get_data()
        self.assertEqual(data_original_df.shape, (10000, 41))
        
    
    """
    
    Testing if the column with not unique values is return false
    
    """
    def test_column_unique_values_path(self):
        pre_process_class = preprocess_module.PreProcessData(query="")
        data_original_df = pre_process_class.get_data()
        self.assertFalse(data_original_df['family_history_3'].is_unique)



if __name__ == '__main__':
    unittest.main()