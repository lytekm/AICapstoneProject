import unittest
from src.data_pipeline import NewsDataPipeline

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = NewsDataPipeline()

    def test_normalization_removes_cbc_junk(self):
        # Sample 'noisy' text similar to what we saw in CBC
        input_text = "Search Search Sign In Quick Links News Being Black in Canada More Actual News Content"
        expected = "Actual News Content"
        
        result = self.pipeline.normalize_text(input_text)
        self.assertEqual(result, expected)

    def test_normalization_removes_ads(self):
        input_text = "Actual News Content Advertisement"
        expected = "Actual News Content"
        
        result = self.pipeline.normalize_text(input_text)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()