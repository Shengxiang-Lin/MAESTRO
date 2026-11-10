# Question construction

`python -m environment generate-questions --config environment/configs/questions.yaml`

The source-code modules use **functional filenames**, while version markers
inside already published Scene formats remain stable for Replay:

| File | Responsibility |
| --- | --- |
| `macro_planning.py` | Current Macro task graph, evidence and frozen goal planning |
| `macro_evidence_world.py` | Typed synthetic evidence and witness mutation |
| `prompt_templates.py` | Question/Scenario author and critic prompts |
| `spec.py` | Scene input contracts and valid reasoning patterns |
| `world.py` | Typed World construction, public source projection |
| `compiler.py` | Compile Macro DAGs and source-scoped executable WorkProducts |
| `evidence_tools.py` | Frozen O1 tool policies and simulated Web Search→Extract chains |
| `stages.py` | Scenario and Question candidate review with durable journals |
| `pipeline.py` | Fixed candidate batch, final strict quality audit, summary |
| `quality.py` | Replay, counterfactual and evidence consistency checks |

The initial `target` is a **candidate count**, not an accepted quota.
Every independently accepted Question is published; no additional candidates
are generated to refill rejected slots. Progress bars show Scenario
Author/Critic, Compile/difficulty audit and Question Author/Critic.
Extra Replay detail is only shown when requested.

The current Macro planning grammar's persisted schema identifier is
`macro_first_v2` for backward-compatible validation. This identifier is
not a Python filename or module name; renaming it inside old data would
break provenance. The earlier Macro-first implementation remains available
for archived-Scene Replay, but is not the default generation path.
