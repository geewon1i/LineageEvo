from lineage_evo.ablation import AblationInput, AblationMode, AblationPolicy
from lineage_evo.prior_fusion import PriorFusionInput, PriorFusionPolicy
from tests.test_search_integration import crossover_prior, global_crossover_prior, global_mutation_prior, mutation_prior


def test_prior_fusion_prefers_local_when_lineage_is_healthy():
    ablation = AblationPolicy().apply(
        AblationInput(AblationMode.OURS_FULL, "mutation", "l1", mutation_prior("improving"), global_mutation_prior())
    )
    context = PriorFusionPolicy().fuse(
        PriorFusionInput(
            ablation_context=ablation,
            lineage_state={"recent_mean_validation_ic_delta": 0.002, "train_validation_ic_gap": 0.001},
        )
    )
    decision = context.prompt_context["fusion_decision"]
    assert decision["local_weight"] > decision["global_weight"]
    assert "Mutation experience for this lineage" in context.prompt_context["rendered_priors"]["lineage_prior_text"]


def test_prior_fusion_prefers_global_for_stagnant_or_biased_lineage():
    prior = mutation_prior("declining")
    prior.bias_risk = "high"
    ablation = AblationPolicy().apply(AblationInput(AblationMode.OURS_FULL, "mutation", "l1", prior, global_mutation_prior()))
    context = PriorFusionPolicy().fuse(
        PriorFusionInput(
            ablation_context=ablation,
            lineage_state={"recent_mean_validation_ic_delta": -0.003, "train_validation_ic_gap": 0.025},
        )
    )
    decision = context.prompt_context["fusion_decision"]
    assert decision["local_weight"] < decision["global_weight"]
    assert "global operator prior" in decision["instruction"]


def test_prior_fusion_ablation_modes_force_weights():
    lineage_only = PriorFusionPolicy().fuse(
        PriorFusionInput(
            AblationPolicy().apply(
                AblationInput(AblationMode.LINEAGE_ONLY, "mutation", "l1", mutation_prior(), global_mutation_prior())
            )
        )
    )
    global_only = PriorFusionPolicy().fuse(
        PriorFusionInput(
            AblationPolicy().apply(
                AblationInput(AblationMode.GLOBAL_ONLY, "mutation", "l1", mutation_prior(), global_mutation_prior())
            )
        )
    )
    assert lineage_only.prompt_context["fusion_decision"]["local_weight"] == 1.0
    assert global_only.prompt_context["fusion_decision"]["global_weight"] == 1.0


def test_prior_fusion_crossover_renders_both_parent_lineage_priors():
    p1 = crossover_prior()
    p1.complementarity_profile = "parent one profile"
    p2 = crossover_prior()
    p2.complementarity_profile = "parent two profile"
    ablation = AblationPolicy().apply(AblationInput(AblationMode.OURS_FULL, "crossover", "l1", p1, global_crossover_prior()))
    context = PriorFusionPolicy().fuse(
        PriorFusionInput(
            ablation_context=ablation,
            parent_lineage_ids=["l1", "l2"],
            parent_lineage_priors=[p1, p2],
        )
    )
    texts = context.prompt_context["rendered_priors"]["parent_lineage_prior_texts"]
    assert "parent one profile" in texts[0]
    assert "parent two profile" in texts[1]
