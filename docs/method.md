# Method and implementation

ReSAIL trains an agent over repeated interaction cycles. ALFWorld and TextCraft provide ordinary observations, actions, rewards, and terminal outcomes. Each trained cycle collects trajectories, constructs any required Experience summaries or teacher targets, runs 30 optimizer updates in `paper` mode, and evaluates its final checkpoint. The next cycle starts from that checkpoint.

## Main methods

| Method | Main behavior |
| --- | --- |
| ReAct | Evaluate the base model without training. |
| RFT | Fine-tune on successful responses from the collected corpus. |
| GRPO | Train the policy with group-relative feedback. |
| EPD | Train with materialized Experience-guided teacher targets. |
| SDPO | Distill from the frozen teacher with ordinary and privileged views. |
| OEL | Distill from new ordinary responses at each update. |
| ReSAIL | Apply SGS, TLB, and PR to OEL. |
| SDPO+ReSAIL | Apply SGS, TLB, and PR to SDPO's cached cycle-start responses. |

The public commands and task/model matrix are in the [experiment catalog](experiment_catalog.md). Methods can share one independently collected C1 corpus for a controlled comparison, or collect method-private fresh C1. Later collection always belongs to the method's own model lineage.

## SGS, TLB, and PR

SGS and TLB form Trajectory-Balanced Selective Distillation (TBSD). PR preserves the student’s privileged-view behavior for the next deployment cycle.

ReSAIL scores eligible response steps, selects the highest-scoring fraction across the full update, and retains information on all source steps. The ALFWorld main setting selects 5% of response steps with retention weight 0.5; TextCraft uses 25% and weight 1.0. Stable ties use trajectory UID and turn index. The selected count is `ceil(ρN)`, raised to the data-parallel floor of eight when needed; there is no trajectory-coverage completion. The loss keeps the original source-trajectory batch size in its outer denominator, including trajectories with no selected step.

For the same ordinary response continuation, the ordinary student's Top-20 token set `S0` defines the support for sensitivity scoring across the two frozen-teacher views and for ordinary distillation. Privileged retention computes its student–teacher KL on a separate Top-20 support `Sc` from the privileged student. Each support also has one residual-tail bucket. These are implementation contracts that matter when comparing numerical results.

The implementation uses the same component names in files, configuration keys, and training metadata:

| Component | Main configuration | Logged metrics |
| --- | --- | --- |
| SGS | `sgs_selection_fraction`, `sgs_score_scope`, `sgs_selection_scope` | `self_distillation/sgs/*` |
| TLB | `tlb_loss_aggregation: trajectory_balanced` | Uses `sdpo_loss_weights` in the shared loss reducer. |
| PR | `pr_weight`, `pr_support`, `pr_kl_direction`, `pr_view` | `pr_loss`, `pr_weighted_loss` |

The `sgs_*` metadata carries scores and selected-step markers; `pr_*` metadata identifies retention rows and their weights. Backend loss evaluation remains in the shared Slime runtime so all methods use the same distributed reducer.

## Code map

| Component | Path |
| --- | --- |
| ALFWorld and TextCraft adapters | [`slime_plugins/agent_tasks/`](../slime_plugins/agent_tasks/) |
| Frozen corpora and shared task mechanics | [`common/frozen/`](../slime_plugins/agent_tasks/common/frozen/) |
| **SGS** — Sensitivity-Guided Selection | [`sgs.py`](../slime_plugins/agent_tasks/common/algorithms/sgs.py): `select_sgs_steps` |
| **TLB** — Trajectory Loss Balancing | [`tlb.py`](../slime_plugins/agent_tasks/common/algorithms/tlb.py): `compute_tlb_weights` |
| **PR** — Privileged Retention | [`pr.py`](../slime_plugins/agent_tasks/common/algorithms/pr.py): `expand_pr_rows` |
| Compose the three components into a training batch | [`resail.py`](../slime_plugins/agent_tasks/common/algorithms/resail.py): `convert_samples_to_train_data` |
| Teacher views and training loss | [`slime/backends/megatron_utils/`](../slime/backends/megatron_utils/) |
| Training and rollout coordination | [`slime/ray/`](../slime/ray/) |
| Main experiment launcher | [`exp/paper/text.py`](../exp/paper/text.py) |

This repository includes the Slime code used by these components.
