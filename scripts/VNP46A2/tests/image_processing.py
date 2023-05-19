from ..image_processing import create_metadata
import numpy as np
import unittest

# import sys

# sys.path.append("../")


class TestCreateMetadata(unittest.TestCase):
    def test_create_metadata(self):
        # Test input values
        array = np.zeros((100, 100), dtype=np.uint8)
        transform = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        driver = "Other"
        nodata = 1
        count = 2
        crs = "test:epsg:4326"

        # Expected output
        expected_metadata = {
            "driver": driver,
            "dtype": array.dtype,
            "nodata": nodata,
            "width": 512,
            "height": 512,
            "count": count,
            "crs": crs,
            "transform": transform,
        }

        # Call the function
        result = create_metadata(array, transform, driver, nodata, count, crs)

        # Assertion
        # assert result == expected_metadata
        self.assertEqual(result, expected_metadata)


if __name__ == "__main__":
    unittest.main()
