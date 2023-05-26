from .context import nightlightsprocessing
import numpy as np
import unittest


class TestExtractQaBits(unittest.TestCase):
    def test_extract_qa_bits(self):
        qa_band = 5
        start_bit = 1
        end_bit = 3

        result = nightlightsprocessing.extract_qa_bits(qa_band, start_bit, end_bit)

        # Work this test out!
        # Not clear why this should be 2
        expected_result = 2

        self.assertEqual(result, expected_result)


class TestCreateMetadata(unittest.TestCase):
    def test_create_metadata(self):
        array = np.zeros((100, 100), dtype=np.uint8)
        transform = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        driver = "Other"
        nodata = 1
        count = 2
        crs = "test:epsg:4326"

        result = nightlightsprocessing.create_metadata(array, transform, driver, nodata, count, crs)

        expected_result = {
            "driver": driver,
            "dtype": array.dtype,
            "nodata": nodata,
            "width": 512,
            "height": 512,
            "count": count,
            "crs": crs,
            "transform": transform,
        }

        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
