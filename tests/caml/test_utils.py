import string

from caml.utils import generate_random_string


class TestGenerateRandomString:
    def test_generate_random_string(self):
        # Test with N = 10
        result = generate_random_string(10)
        assert len(result) == 10  # Check if the length of the result is 10
        assert all(
            c in string.ascii_lowercase + string.digits for c in result
        )  # Check if the result contains only lowercase letters and digits

        # Test with N = 5
        result = generate_random_string(5)
        assert len(result) == 5  # Check if the length of the result is 5
        assert all(
            c in string.ascii_lowercase + string.digits for c in result
        )  # Check if the result contains only lowercase letters and digits

        # Test with N = 0
        result = generate_random_string(0)
        assert len(result) == 0  # Check if the length of the result is 0

        # Test with N = -5
        result = generate_random_string(-5)
        assert len(result) == 0  # Check if the length of the result is 0
