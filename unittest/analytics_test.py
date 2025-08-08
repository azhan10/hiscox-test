import unittest
import sys, pathlib

from pathlib import Path

sys.path.append(str(pathlib.Path().resolve().parent) + "/scripts/")

import analytics as analytics_module


"""

I could add more tests like:
    - Testing the correlation value (columns with no randomness) between two factors
    - Testing the number of categorical columns
    - Testing the pairwise is boolean true

"""

class TestAnalytics(unittest.TestCase):

    
    """
    
    Testing if the data path is correct
    
    """
    def test_data_path(self):
        analytics_class = analytics_module.Analytics(data=[])
        self.assertEqual(analytics_class.data_path(), str(pathlib.Path().resolve().parent) + "/data/")


if __name__ == '__main__':
    unittest.main()