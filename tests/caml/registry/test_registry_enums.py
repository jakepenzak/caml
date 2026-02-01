from caml.registry import EstimatorFamily, ScorerFamily


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


class TestScorerFamily:
    """Tests for ScorerFamily enum."""

    def test_enum_values(self):
        """Test that ScorerFamily has expected values."""
        assert ScorerFamily.ORACLE.value == "oracle"
        assert ScorerFamily.PLUG_IN.value == "plug_in"
        assert ScorerFamily.PSUEDO_OUTCOME.value == "pseudo_outcome"
        assert ScorerFamily.RANKING_RELATIVE_PROXY.value == "ranking_relative_proxy"
        assert ScorerFamily.RANKING_CURVE.value == "ranking_curve"
        assert ScorerFamily.POLICY.value == "policy"
        assert ScorerFamily.CUSTOM.value == "custom"

    def test_enum_from_string(self):
        """Test creating ScorerFamily from string."""
        assert ScorerFamily("oracle") == ScorerFamily.ORACLE
        assert ScorerFamily("plug_in") == ScorerFamily.PLUG_IN
        assert ScorerFamily("pseudo_outcome") == ScorerFamily.PSUEDO_OUTCOME
        assert (
            ScorerFamily("ranking_relative_proxy")
            == ScorerFamily.RANKING_RELATIVE_PROXY
        )
        assert ScorerFamily("ranking_curve") == ScorerFamily.RANKING_CURVE
        assert ScorerFamily("policy") == ScorerFamily.POLICY
        assert ScorerFamily("custom") == ScorerFamily.CUSTOM

    def test_enum_membership(self):
        """Test ScorerFamily enum membership."""
        all_families = {
            ScorerFamily.ORACLE,
            ScorerFamily.PLUG_IN,
            ScorerFamily.PSUEDO_OUTCOME,
            ScorerFamily.RANKING_RELATIVE_PROXY,
            ScorerFamily.RANKING_CURVE,
            ScorerFamily.POLICY,
            ScorerFamily.CUSTOM,
        }
        assert len(all_families) == 7
        assert ScorerFamily.ORACLE in ScorerFamily
        assert ScorerFamily.CUSTOM in ScorerFamily
