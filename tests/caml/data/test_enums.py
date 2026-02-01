"""Tests for caml.data.data_enums."""

import pytest

from caml.data import Estimand, OutcomeType, TreatmentType

pytestmark = [pytest.mark.data]


def test_treatment_type_is_discrete():
    """Test TreatmentType.is_discrete() method."""
    assert TreatmentType.BINARY.is_discrete()
    assert TreatmentType.MULTI.is_discrete()
    assert not TreatmentType.CONTINUOUS.is_discrete()


def test_outcome_type_is_discrete():
    """Test OutcomeType.is_discrete() method."""
    assert OutcomeType.BINARY.is_discrete()
    assert not OutcomeType.CONTINUOUS.is_discrete()


def test_estimand_values_are_unique_strings():
    """Test that all Estimand values are unique strings."""
    values = [estimand.value for estimand in Estimand]
    assert all(isinstance(value, str) for value in values)
    assert len(values) == len(set(values))


def test_treatment_type_values():
    """Test TreatmentType enum values."""
    assert TreatmentType.BINARY.value == "binary"
    assert TreatmentType.MULTI.value == "multi"
    assert TreatmentType.CONTINUOUS.value == "continuous"


def test_outcome_type_values():
    """Test OutcomeType enum values."""
    assert OutcomeType.BINARY.value == "binary"
    assert OutcomeType.CONTINUOUS.value == "continuous"


def test_estimand_values():
    """Test Estimand enum values."""
    assert Estimand.ATE.value == "ate"
    assert Estimand.ATT.value == "att"
    assert Estimand.ATC.value == "atc"
    assert Estimand.CATE.value == "cate"
    assert Estimand.GATE.value == "gate"


def test_treatment_type_count():
    """Test that TreatmentType has exactly 3 members."""
    assert len(TreatmentType) == 3


def test_outcome_type_count():
    """Test that OutcomeType has exactly 2 members."""
    assert len(OutcomeType) == 2


def test_estimand_count():
    """Test that Estimand has exactly 5 members."""
    assert len(Estimand) == 5
