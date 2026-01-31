from caml.registry.registry_schema import EstimatorFamily


class TestEstimatorFamily:
    """Tests for EstimatorFamily enum."""

    def test_enum_values(self):
        """Test that EstimatorFamily has expected values."""
        assert EstimatorFamily.DML.value == "dml"
        assert EstimatorFamily.DR.value == "dr"
        assert EstimatorFamily.META.value == "meta"
        assert EstimatorFamily.ORF.value == "orf"
        assert EstimatorFamily.CUSTOM.value == "custom"

    def test_enum_from_string(self):
        """Test creating EstimatorFamily from string."""
        assert EstimatorFamily("dml") == EstimatorFamily.DML
        assert EstimatorFamily("dr") == EstimatorFamily.DR
        assert EstimatorFamily("meta") == EstimatorFamily.META
        assert EstimatorFamily("orf") == EstimatorFamily.ORF
        assert EstimatorFamily("custom") == EstimatorFamily.CUSTOM

    def test_enum_membership(self):
        """Test EstimatorFamily enum membership."""
        all_families = {
            EstimatorFamily.DML,
            EstimatorFamily.DR,
            EstimatorFamily.META,
            EstimatorFamily.ORF,
            EstimatorFamily.CUSTOM,
        }
        assert len(all_families) == 5
        assert EstimatorFamily.DML in EstimatorFamily
        assert EstimatorFamily.CUSTOM in EstimatorFamily
