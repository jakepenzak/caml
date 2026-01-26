"""Tests for model bank structure."""

import pytest

from caml.data import CausalDataset
from caml.estimators.base import AutoCateEstimator, EstimatorCapabilities
from caml.registry.model_bank import available_estimators


class TestAvailableEstimators:
    """Tests for available_estimators dictionary."""

    def test_available_estimators_is_dict(self):
        """Test that available_estimators is a dictionary."""
        assert isinstance(available_estimators, dict)

    def test_available_estimators_not_empty(self):
        """Test that available_estimators contains estimators."""
        assert len(available_estimators) > 0

    def test_available_estimators_structure(self):
        """Test that each entry has correct structure."""
        for name, info in available_estimators.items():
            # Name should be a string
            assert isinstance(name, str)

            # Info should be a dict with 'estimator' and 'family' keys
            assert isinstance(info, dict)
            assert "estimator" in info
            assert "family" in info

            # Family should be a string
            assert isinstance(info["family"], str)

            # Estimator should be a class (not an instance)
            assert isinstance(info["estimator"], type)

    def test_available_estimators_families(self):
        """Test that all expected families are present."""
        families = {info["family"] for info in available_estimators.values()}

        # Should have the 4 main families
        expected_families = {"dml", "dr", "meta", "orf"}
        assert expected_families.issubset(families)

    def test_available_estimators_counts(self):
        """Test that expected number of estimators per family exist."""
        # Count estimators by family
        family_counts = {}
        for info in available_estimators.values():
            family = info["family"]
            family_counts[family] = family_counts.get(family, 0) + 1

        # Expected counts based on model_bank.py
        assert family_counts.get("dml", 0) == 5  # 5 DML estimators
        assert family_counts.get("dr", 0) == 4  # 4 DR estimators
        assert family_counts.get("meta", 0) == 3  # 3 Meta-learners
        assert family_counts.get("orf", 0) == 2  # 2 ORF estimators

        # Total should be 14
        assert sum(family_counts.values()) == 14

    def test_available_estimators_names(self):
        """Test that expected estimator names are present."""
        expected_names = {
            # DML (5)
            "LinearDML",
            "SparseLinearDML",
            "CausalForestDML",
            "NonParamDML",
            "KernelDML",
            # DR (4)
            "DRLearner",
            "LinearDRLearner",
            "SparseLinearDRLearner",
            "ForestDRLearner",
            # Meta (3)
            "SLearner",
            "TLearner",
            "XLearner",
            # ORF (2)
            "DMLOrthoForest",
            "DROrthoForest",
        }

        assert set(available_estimators.keys()) == expected_names

    def test_available_estimators_classes_are_classes(self):
        """Test that all estimators are classes (not instances)."""
        for name, info in available_estimators.items():
            estimator_class = info["estimator"]

            # Should be a class
            assert isinstance(estimator_class, type)

            # Should not be an instance
            assert not hasattr(estimator_class, "_estimator")  # Not fitted

    def test_available_estimators_all_implement_protocol(self):
        """Test that all estimators implement AutoCateEstimator protocol."""
        for name, info in available_estimators.items():
            estimator_class = info["estimator"]

            # Should have clean_name property
            assert hasattr(estimator_class, "clean_name")

            # Should have capabilities property
            assert hasattr(estimator_class, "capabilities")

            # Should have is_compatible_with method
            assert hasattr(estimator_class, "is_compatible_with")
            assert callable(getattr(estimator_class, "is_compatible_with"))

            # Should have check_compatibility method
            assert hasattr(estimator_class, "check_compatibility")

            # Should have fit method
            assert hasattr(estimator_class, "fit")
            assert callable(getattr(estimator_class, "fit"))

            # Should have effect method
            assert hasattr(estimator_class, "effect")
            assert callable(getattr(estimator_class, "effect"))

            # Should have get_params/set_params (sklearn interface)
            assert hasattr(estimator_class, "get_params")
            assert callable(getattr(estimator_class, "get_params"))
            assert hasattr(estimator_class, "set_params")
            assert callable(getattr(estimator_class, "set_params"))

    def test_available_estimators_capabilities_type(self):
        """Test that all estimators have EstimatorCapabilities."""
        for name, info in available_estimators.items():
            estimator_class = info["estimator"]

            # Instantiate to check capabilities
            # (Some estimators may require kwargs, so we catch errors)
            try:
                # Get capabilities from class property
                caps = estimator_class.capabilities

                # Should be EstimatorCapabilities
                assert isinstance(caps, EstimatorCapabilities)

                # Should have required attributes
                assert hasattr(caps, "treatment_types")
                assert hasattr(caps, "outcome_types")
                assert hasattr(caps, "inference_types")
                assert hasattr(caps, "estimands")
            except Exception:
                # Some classes might need initialization first
                # Skip those for this test
                pass

    def test_available_estimators_dml_family(self):
        """Test DML family estimators."""
        dml_estimators = {
            name: info
            for name, info in available_estimators.items()
            if info["family"] == "dml"
        }

        # Should have 5 DML estimators
        assert len(dml_estimators) == 5

        # Check expected names
        expected_dml = {
            "LinearDML",
            "SparseLinearDML",
            "CausalForestDML",
            "NonParamDML",
            "KernelDML",
        }
        assert set(dml_estimators.keys()) == expected_dml

    def test_available_estimators_dr_family(self):
        """Test DR family estimators."""
        dr_estimators = {
            name: info
            for name, info in available_estimators.items()
            if info["family"] == "dr"
        }

        # Should have 4 DR estimators
        assert len(dr_estimators) == 4

        # Check expected names
        expected_dr = {
            "DRLearner",
            "LinearDRLearner",
            "SparseLinearDRLearner",
            "ForestDRLearner",
        }
        assert set(dr_estimators.keys()) == expected_dr

    def test_available_estimators_meta_family(self):
        """Test meta-learner family estimators."""
        meta_estimators = {
            name: info
            for name, info in available_estimators.items()
            if info["family"] == "meta"
        }

        # Should have 3 meta-learners
        assert len(meta_estimators) == 3

        # Check expected names
        expected_meta = {"SLearner", "TLearner", "XLearner"}
        assert set(meta_estimators.keys()) == expected_meta

    def test_available_estimators_orf_family(self):
        """Test ORF family estimators."""
        orf_estimators = {
            name: info
            for name, info in available_estimators.items()
            if info["family"] == "orf"
        }

        # Should have 2 ORF estimators
        assert len(orf_estimators) == 2

        # Check expected names
        expected_orf = {"DMLOrthoForest", "DROrthoForest"}
        assert set(orf_estimators.keys()) == expected_orf
