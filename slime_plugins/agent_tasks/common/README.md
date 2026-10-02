# Shared agent-task components

ALFWorld and TextCraft use these modules for trajectory collection, training inputs, and evaluation:

| Path | Purpose |
| --- | --- |
| `data_source.py` | Construct trajectory batches. |
| `frozen/` | Load collected corpora, prepare teacher targets, and record evaluation results. |
| `algorithms/sgs.py` | Select response steps by privileged-information sensitivity. |
| `algorithms/tlb.py` | Balance losses across source trajectories. |
| `algorithms/pr.py` | Construct privileged-retention training rows. |
| `algorithms/resail.py` | Combine these components into training inputs. |
| `eval.py` | Handle evaluation seeds and replicate names. |
| `logging.py` | Aggregate task metrics. |

See the [method guide](../../../docs/method.md) for configuration keys and the [run guide](../../../docs/reproduction.md) for experiment commands and result files.
