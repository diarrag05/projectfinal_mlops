import unittest
from src.data_extraction import load_data
import pandas as pd

class TestDataExtraction(unittest.TestCase):
    def test_load_data_success(self):
        data = load_data('dataset.csv')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('text', data.columns)
        self.assertIn('label', data.columns)

    def test_load_data_file_not_found(self):
        data = load_data('inexistant.csv')
        self.assertIsNone(data)

if __name__ == '__main__':
    unittest.main()
