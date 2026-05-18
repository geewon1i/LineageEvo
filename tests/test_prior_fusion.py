from lineage_evo.prior_fusion import FusionMode, PriorFusionInput, PriorFusionPolicy
from tests.test_search_integration import global_mutation_prior, mutation_prior


def test_prior_fusion_ours_full_includes_lineage_and_global():
    context = PriorFusionPolicy().fuse(
        PriorFusionInput(
            mode=FusionMode.OURS_FULL,
            operator="mutation",
            lineage_id="l1",
            lineage_prior=mutation_prior(),
            global_prior=global_mutation_prior(),
        )
    )
    assert "lineage_prior" in context.prompt_context
    assert "global_prior" in context.prompt_context
    assert context.prior_updates_enabled is True


def test_prior_fusion_ablation_modes_exclude_expected_context():
    lineage_only = PriorFusionPolicy().fuse(
        PriorFusionInput(FusionMode.LINEAGE_ONLY, "mutation", "l1", mutation_prior(), global_mutation_prior())
    )
    global_only = PriorFusionPolicy().fuse(
        PriorFusionInput(FusionMode.GLOBAL_ONLY, "mutation", "l1", mutation_prior(), global_mutation_prior())
    )
    no_update = PriorFusionPolicy().fuse(
        PriorFusionInput(FusionMode.NO_PRIOR_UPDATE, "mutation", "l1", mutation_prior(), global_mutation_prior())
    )
    assert "lineage_prior" in lineage_only.prompt_context
    assert "global_prior" not in lineage_only.prompt_context
    assert "global_prior" in global_only.prompt_context
    assert "lineage_prior" not in global_only.prompt_context
    assert no_update.prior_updates_enabled is False


def test_shuffled_lineage_prior_uses_other_lineage():
    context = PriorFusionPolicy().fuse(
        PriorFusionInput(
            mode=FusionMode.SHUFFLED_LINEAGE_PRIOR,
            operator="mutation",
            lineage_id="l1",
            lineage_prior=mutation_prior("own"),
            global_prior=global_mutation_prior(),
            all_lineage_priors={"l1": mutation_prior("own"), "l2": mutation_prior("other")},
        )
    )
    assert context.prompt_context["lineage_prior"]["quality_trend"] == "other"

