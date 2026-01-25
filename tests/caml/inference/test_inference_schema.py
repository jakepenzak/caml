"""Tests for caml.inference.inference_schema."""

import pytest

from caml.inference.inference_schema import InferenceType


class TestInferenceType:
    """Tests for InferenceType enum."""

    def test_analytic_value(self):
        """Test ANALYTIC has correct value."""
        assert InferenceType.ANALYTIC.value == "analytic"

    def test_bootstrap_value(self):
        """Test BOOTSTRAP has correct value."""
        assert InferenceType.BOOTSTRAP.value == "bootstrap"

    def test_all_members_exist(self):
        """Test all expected enum members exist."""
        members = {member.name for member in InferenceType}
        expected = {"ANALYTIC", "BOOTSTRAP"}
        assert members == expected

    def test_enum_equality(self):
        """Test enum member equality."""
        assert InferenceType.ANALYTIC == InferenceType.ANALYTIC
        assert InferenceType.ANALYTIC != InferenceType.BOOTSTRAP

    def test_enum_from_string(self):
        """Test constructing enum from string value."""
        assert InferenceType("analytic") == InferenceType.ANALYTIC
        assert InferenceType("bootstrap") == InferenceType.BOOTSTRAP

    def test_invalid_string_raises(self):
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError):
            InferenceType("invalid")

    def test_enum_iteration(self):
        """Test iterating over enum members."""
        members = list(InferenceType)
        assert len(members) == 2
        assert InferenceType.ANALYTIC in members
        assert InferenceType.BOOTSTRAP in members
