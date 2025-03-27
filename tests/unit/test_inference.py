import unittest
from src.inference import predict

class TestInference(unittest.TestCase):
    def test_predict(self):
        result = predict("I love using this app!")
        self.assertIn(result, ["Positif", "Très positif", "Négatif", "Très négatif", "Neutre"])

if __name__ == '__main__':
    unittest.main()
