"""Tests for caml/registry/model_bank.py."""

import pytest

from caml.estimators.wrappers.dml import (
    WrappedCausalForestDML,
    WrappedKernelDML,
    WrappedLinearDML,
    WrappedNonParamDML,
    WrappedSparseLinearDML,
)
from caml.estimators.wrappers.dr import (
    WrappedDRLearner,
    WrappedForestDRLearner,
    WrappedLinearDRLearner,
    WrappedSparseLinearDRLearner,
)
from caml.estimators.wrappers.meta import (
    WrappedSLearner,
    WrappedTLearner,
    WrappedXLearner,
)
from caml.estimators.wrappers.orf import WrappedDMLOrthoForest, WrappedDROrthoForest
from caml.registry.model_bank import available_estimators

pytestmark = pytest.mark.registry


class TestModelBank:
    """Tests for the available_estimators dictionary."""

    def test_available_estimators_is_dict(self):
        """Test that available_estimators is a dictionary."""
        assert isinstance(available_estimators, dict)

    def test_available_estimators_has_expected_count(self):
        """Test that available_estimators has 14 estimators."""
        assert len(available_estimators) == 14

    def test_all_entries_have_required_keys(self):
        """Test that all entries have 'estimator' and 'family' keys."""
        for name, entry in available_estimators.items():
            assert "estimator" in entry, f"{name} missing 'estimator' key"
            assert "family" in entry, f"{name} missing 'family' key"

    def test_all_families_are_valid(self):
        """Test that all families are one of: dml, dr, meta, orf."""
        valid_families = {"dml", "dr", "meta", "orf"}
        for name, entry in available_estimators.items():
            assert (
                entry["family"] in valid_families
            ), f"{name} has invalid family: {entry['family']}"

    def test_dml_family_estimators(self):
        """Test that DML family has 5 estimators."""
        dml_estimators = {
            name: entry
            for name, entry in available_estimators.items()
            if entry["family"] == "dml"
        }
        assert len(dml_estimators) == 5
        expected_names = {
            "CausalForestDML",
            "KernelDML",
            "LinearDML",
            "NonParamDML",
            "SparseLinearDML",
        }
        assert set(dml_estimators.keys()) == expected_names

    def test_dr_family_estimators(self):
        """Test that DR family has 4 estimators."""
        dr_estimators = {
            name: entry
            for name, entry in available_estimators.items()
            if entry["family"] == "dr"
        }
        assert len(dr_estimators) == 4
        expected_names = {
            "DRLearner",
            "ForestDRLearner",
            "LinearDRLearner",
            "SparseLinearDRLearner",
        }
        assert set(dr_estimators.keys()) == expected_names

    def test_meta_family_estimators(self):
        """Test that meta family has 3 estimators."""
        meta_estimators = {
            name: entry
            for name, entry in available_estimators.items()
            if entry["family"] == "meta"
        }
        assert len(meta_estimators) == 3
        expected_names = {"SLearner", "TLearner", "XLearner"}
        assert set(meta_estimators.keys()) == expected_names

    def test_orf_family_estimators(self):
        """Test that ORF family has 2 estimators."""
        orf_estimators = {
            name: entry
            for name, entry in available_estimators.items()
            if entry["family"] == "orf"
        }
        assert len(orf_estimators) == 2
        expected_names = {"DMLOrthoForest", "DROrthoForest"}
        assert set(orf_estimators.keys()) == expected_names

    def test_estimator_classes_are_correct(self):
        """Test that estimator classes match expected wrapper classes."""
        expected_classes = {
            "CausalForestDML": WrappedCausalForestDML,
            "KernelDML": WrappedKernelDML,
            "LinearDML": WrappedLinearDML,
            "NonParamDML": WrappedNonParamDML,
            "SparseLinearDML": WrappedSparseLinearDML,
            "DRLearner": WrappedDRLearner,
            "ForestDRLearner": WrappedForestDRLearner,
            "LinearDRLearner": WrappedLinearDRLearner,
            "SparseLinearDRLearner": WrappedSparseLinearDRLearner,
            "SLearner": WrappedSLearner,
            "TLearner": WrappedTLearner,
            "XLearner": WrappedXLearner,
            "DMLOrthoForest": WrappedDMLOrthoForest,
            "DROrthoForest": WrappedDROrthoForest,
        }
        for name, expected_class in expected_classes.items():
            assert available_estimators[name]["estimator"] == expected_class

    def test_all_estimator_classes_are_callable(self):
        """Test that all estimator classes can be instantiated."""
        for name, entry in available_estimators.items():
            estimator_class = entry["estimator"]
            assert callable(estimator_class), f"{name} estimator is not callable"

    def test_estimator_names_match_wrapper_classes(self):
        """Test that estimator names are consistent with their class names."""
        expected_mappings = {
            "CausalForestDML": "WrappedCausalForestDML",
            "KernelDML": "WrappedKernelDML",
            "LinearDML": "WrappedLinearDML",
            "NonParamDML": "WrappedNonParamDML",
            "SparseLinearDML": "WrappedSparseLinearDML",
            "DRLearner": "WrappedDRLearner",
            "ForestDRLearner": "WrappedForestDRLearner",
            "LinearDRLearner": "WrappedLinearDRLearner",
            "SparseLinearDRLearner": "WrappedSparseLinearDRLearner",
            "SLearner": "WrappedSLearner",
            "TLearner": "WrappedTLearner",
            "XLearner": "WrappedXLearner",
            "DMLOrthoForest": "WrappedDMLOrthoForest",
            "DROrthoForest": "WrappedDROrthoForest",
        }
        for name, entry in available_estimators.items():
            class_name = entry["estimator"].__name__
            assert class_name == expected_mappings[name]
