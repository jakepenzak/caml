import string
import unittest

from caml.utils import generate_random_string


class TestGenerateRandomString(unittest.TestCase):
    def test_generate_random_string(self):
        # Test with N = 10
        result = generate_random_string(10)
        self.assertEqual(len(result), 10)  # Check if the length of the result is 10
        self.assertTrue(
            all(c in string.ascii_lowercase + string.digits for c in result)
        )  # Check if the result contains only lowercase letters and digits

        # Test with N = 5
        result = generate_random_string(5)
        self.assertEqual(len(result), 5)  # Check if the length of the result is 5
        self.assertTrue(
            all(c in string.ascii_lowercase + string.digits for c in result)
        )  # Check if the result contains only lowercase letters and digits

        # Test with N = 0
        result = generate_random_string(0)
        self.assertEqual(len(result), 0)  # Check if the length of the result is 0

        # Test with N = -5
        result = generate_random_string(-5)
        self.assertEqual(len(result), 0)  # Check if the length of the result is 0


if __name__ == "__main__":
    unittest.main()
