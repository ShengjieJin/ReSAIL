# Main experiment catalog

Use [`scripts/experiments/text_cycles.sh`](../scripts/experiments/text_cycles.sh) for every main run. It accepts `--task alfworld|textcraft`, `--model 4b|8b`, and one of the eight methods below. The [reproduction guide](reproduction.md) gives guarded host commands; the [configuration guide](../configs/main/text/README.md) lists the full task/model matrix.

| Method | Run design |
| --- | --- |
| `react` | One evaluation of the base model; no training cycle. |
| `rft` | Three cycles of successful-response fine-tuning. |
| `grpo` | Three cycles of group-relative policy training. |
| `epd` | Three cycles with materialized Experience-guided targets. |
| `sdpo` | Three cycles of teacher distillation. |
| `oel` | Three cycles using new ordinary responses at each update. |
| `resail` | Three OEL-based cycles with sensitivity selection and privileged retention. |
| `sdpo_resail` | Three SDPO-based cycles with the same selection and retention. |

Every trained cycle has 30 optimizer updates in `paper` mode and evaluates its final checkpoint. Training seed 1234, collection seeds 42/43/44, and evaluation seeds 314159/314160/314161 are configured in the public launcher. ALFWorld evaluates ID and OOD task panels; TextCraft evaluates the official task-ID split. The main ALFWorld ReSAIL setting selects the global Top-5% of response steps with retention weight 0.5; TextCraft uses Top-25% with weight 1.0. See [method details](method.md).

## C1 inputs

- **Shared:** prepare one verified corpus with `scripts/prepare/text_c1.sh --task TASK --model SIZE --c1-input PATH`, then pass that same path to each `text_cycles.sh` command. This controls the initial input across methods of one task and model size.
- **Fresh per method:** omit `--c1-input` from `text_cycles.sh`; that method collects its own C1 corpus. This is the shortest route for one run from scratch.

Both modes use the live task environment. Shared C1 contains 960 ALFWorld or 240 TextCraft trajectories. Each method collects its own C2 and C3 data from its preceding model. The shared path is read-only during training; method outputs remain under unique run roots.

The source package includes the main launcher, collection/guidance/EPD helpers, and task adapters. Download model weights and task caches separately; generated inputs, checkpoints, and run outputs stay local.
