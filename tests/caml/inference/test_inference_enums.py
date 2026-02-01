"""Tests for caml.inference.inference_enums module."""

import pytest

from caml.inference import InferenceType

pytestmark = pytest.mark.inference


def test_inference_type_values():
    """Test InferenceType enum values."""
    assert InferenceType.ANALYTIC.value == "analytic"
    assert InferenceType.BOOTSTRAP.value == "bootstrap"


def test_inference_type_count():
    """Test that InferenceType has exactly 2 members."""
    assert len(InferenceType) == 2


def test_inference_type_members():
    """Test that all expected InferenceType members exist."""
    assert hasattr(InferenceType, "ANALYTIC")
    assert hasattr(InferenceType, "BOOTSTRAP")


def test_inference_type_values_are_unique_strings():
    """Test that all InferenceType values are unique strings."""
    values = [inf_type.value for inf_type in InferenceType]
    assert all(isinstance(value, str) for value in values)
    assert len(values) == len(set(values))


def test_inference_type_iteration():
    """Test that InferenceType can be iterated."""
    types = list(InferenceType)
    assert len(types) == 2
    assert InferenceType.ANALYTIC in types
    assert InferenceType.BOOTSTRAP in types


def test_inference_type_membership():
    """Test membership checking for InferenceType."""
    assert InferenceType.ANALYTIC in InferenceType
    assert InferenceType.BOOTSTRAP in InferenceType
