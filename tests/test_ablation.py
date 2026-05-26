from lineage_evo.ablation import AblationInput, AblationMode, AblationPolicy
from tests.test_search_integration import global_mutation_prior, mutation_prior


def test_ablation_ours_full_keeps_lineage_and_global():
    context = AblationPolicy().apply(
        AblationInput(
            mode=AblationMode.OURS_FULL,
            operator="mutation",
            lineage_id="l1",
            lineage_prior=mutation_prior(),
            global_prior=global_mutation_prior(),
        )
    )
    assert context.lineage_prior is not None
    assert context.global_prior is not None
    assert context.prior_updates_enabled is True


def test_ablation_modes_exclude_expected_sources():
    lineage_only = AblationPolicy().apply(
        AblationInput(AblationMode.LINEAGE_ONLY, "mutation", "l1", mutation_prior(), global_mutation_prior())
    )
    global_only = AblationPolicy().apply(
        AblationInput(AblationMode.GLOBAL_ONLY, "mutation", "l1", mutation_prior(), global_mutation_prior())
    )
    no_update = AblationPolicy().apply(
        AblationInput(AblationMode.NO_PRIOR_UPDATE, "mutation", "l1", mutation_prior(), global_mutation_prior())
    )
    assert lineage_only.lineage_prior is not None
    assert lineage_only.global_prior is None
    assert global_only.global_prior is not None
    assert global_only.lineage_prior is None
    assert no_update.prior_updates_enabled is False


def test_shuffled_lineage_prior_uses_other_lineage():
    context = AblationPolicy().apply(
        AblationInput(
            mode=AblationMode.SHUFFLED_LINEAGE_PRIOR,
            operator="mutation",
            lineage_id="l1",
            lineage_prior=mutation_prior("own"),
            global_prior=global_mutation_prior(),
            all_lineage_priors={"l1": mutation_prior("own"), "l2": mutation_prior("other")},
        )
    )
    assert context.lineage_prior.hint == "other"
    assert context.shuffled_from_lineage == "l2"
