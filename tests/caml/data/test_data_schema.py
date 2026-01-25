"""Tests for caml.data.data_schema enums."""

import pytest

from caml.data.data_schema import Estimand, OutcomeType, TreatmentType


class TestTreatmentType:
    """Tests for TreatmentType enum."""

    def test_binary_value(self):
        """Test BINARY has correct value."""
        assert TreatmentType.BINARY.value == "binary"

    def test_multi_value(self):
        """Test MULTI has correct value."""
        assert TreatmentType.MULTI.value == "multi"

    def test_continuous_value(self):
        """Test CONTINUOUS has correct value."""
        assert TreatmentType.CONTINUOUS.value == "continuous"

    def test_binary_is_discrete(self):
        """Test BINARY is classified as discrete."""
        assert TreatmentType.BINARY.is_discrete() is True

    def test_multi_is_discrete(self):
        """Test MULTI is classified as discrete."""
        assert TreatmentType.MULTI.is_discrete() is True

    def test_continuous_not_discrete(self):
        """Test CONTINUOUS is not classified as discrete."""
        assert TreatmentType.CONTINUOUS.is_discrete() is False

    def test_all_members_exist(self):
        """Test all expected enum members exist."""
        members = {member.name for member in TreatmentType}
        expected = {"BINARY", "MULTI", "CONTINUOUS"}
        assert members == expected

    def test_enum_equality(self):
        """Test enum member equality."""
        assert TreatmentType.BINARY == TreatmentType.BINARY
        assert TreatmentType.BINARY != TreatmentType.MULTI

    def test_enum_from_string(self):
        """Test constructing enum from string value."""
        assert TreatmentType("binary") == TreatmentType.BINARY
        assert TreatmentType("multi") == TreatmentType.MULTI
        assert TreatmentType("continuous") == TreatmentType.CONTINUOUS

    def test_invalid_string_raises(self):
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError):
            TreatmentType("invalid")


class TestOutcomeType:
    """Tests for OutcomeType enum."""

    def test_binary_value(self):
        """Test BINARY has correct value."""
        assert OutcomeType.BINARY.value == "binary"

    def test_continuous_value(self):
        """Test CONTINUOUS has correct value."""
        assert OutcomeType.CONTINUOUS.value == "continuous"

    def test_binary_is_discrete(self):
        """Test BINARY is classified as discrete."""
        assert OutcomeType.BINARY.is_discrete() is True

    def test_continuous_not_discrete(self):
        """Test CONTINUOUS is not classified as discrete."""
        assert OutcomeType.CONTINUOUS.is_discrete() is False

    def test_all_members_exist(self):
        """Test all expected enum members exist."""
        members = {member.name for member in OutcomeType}
        expected = {"BINARY", "CONTINUOUS"}
        assert members == expected

    def test_enum_equality(self):
        """Test enum member equality."""
        assert OutcomeType.BINARY == OutcomeType.BINARY
        assert OutcomeType.BINARY != OutcomeType.CONTINUOUS

    def test_enum_from_string(self):
        """Test constructing enum from string value."""
        assert OutcomeType("binary") == OutcomeType.BINARY
        assert OutcomeType("continuous") == OutcomeType.CONTINUOUS

    def test_invalid_string_raises(self):
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError):
            OutcomeType("invalid")


class TestEstimand:
    """Tests for Estimand enum."""

    def test_ate_value(self):
        """Test ATE has correct value."""
        assert Estimand.ATE.value == "ate"

    def test_att_value(self):
        """Test ATT has correct value."""
        assert Estimand.ATT.value == "att"

    def test_atc_value(self):
        """Test ATC has correct value."""
        assert Estimand.ATC.value == "atc"

    def test_cate_value(self):
        """Test CATE has correct value."""
        assert Estimand.CATE.value == "cate"

    def test_gate_value(self):
        """Test GATE has correct value."""
        assert Estimand.GATE.value == "gate"

    def test_all_members_exist(self):
        """Test all expected enum members exist."""
        members = {member.name for member in Estimand}
        expected = {"ATE", "ATT", "ATC", "CATE", "GATE"}
        assert members == expected

    def test_enum_equality(self):
        """Test enum member equality."""
        assert Estimand.ATE == Estimand.ATE
        assert Estimand.ATE != Estimand.ATT

    def test_enum_from_string(self):
        """Test constructing enum from string value."""
        assert Estimand("ate") == Estimand.ATE
        assert Estimand("att") == Estimand.ATT
        assert Estimand("atc") == Estimand.ATC
        assert Estimand("cate") == Estimand.CATE
        assert Estimand("gate") == Estimand.GATE

    def test_invalid_string_raises(self):
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError):
            Estimand("invalid")


class TestCrossEnumComparison:
    """Tests for comparisons across different enum types."""

    def test_treatment_type_not_equal_outcome_type(self):
        """Test TreatmentType members not equal to OutcomeType members."""
        assert TreatmentType.BINARY != OutcomeType.BINARY
        assert TreatmentType.CONTINUOUS != OutcomeType.CONTINUOUS

    def test_different_enums_not_equal_despite_same_value(self):
        """Test enums with same string value are not equal across types."""
        # Both have value "binary" but different types
        assert TreatmentType.BINARY.value == OutcomeType.BINARY.value
        assert TreatmentType.BINARY != OutcomeType.BINARY
