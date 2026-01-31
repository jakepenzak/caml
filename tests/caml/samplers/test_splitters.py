"""Tests for caml.samplers.splitters module."""

import numpy as np
import pytest
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from caml.samplers import create_splitter

pytestmark = [pytest.mark.samplers]


class TestCreateSplitter:
    """Test create_splitter factory function."""

    def test_default_returns_kfold(self):
        """Test that default parameters return KFold splitter."""
        splitter = create_splitter()

        assert isinstance(splitter, KFold)
        assert splitter.n_splits == 3

    def test_custom_cv_folds(self):
        """Test that cv parameter sets number of folds."""
        splitter = create_splitter(cv=5)

        assert isinstance(splitter, KFold)
        assert splitter.n_splits == 5

    def test_stratified_without_groups(self):
        """Test that stratified=True returns StratifiedKFold."""
        splitter = create_splitter(stratified=True)

        assert isinstance(splitter, StratifiedKFold)

    def test_groups_without_stratified(self):
        """Test that groups returns GroupKFold."""
        groups = np.array([0, 0, 1, 1, 2, 2])
        splitter = create_splitter(groups=groups)

        assert isinstance(splitter, GroupKFold)

    def test_groups_with_stratified(self):
        """Test that groups with stratified returns StratifiedGroupKFold."""
        groups = np.array([0, 0, 1, 1, 2, 2])
        splitter = create_splitter(groups=groups, stratified=True)

        assert isinstance(splitter, StratifiedGroupKFold)

    def test_random_state_deterministic(self):
        """Test that random_state produces deterministic splits."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        splitter1 = create_splitter(cv=3, random_state=42)
        splitter2 = create_splitter(cv=3, random_state=42)

        splits1 = list(splitter1.split(X, y))
        splits2 = list(splitter2.split(X, y))

        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)
