# MAESTRO pipeline configuration

Edit paths and model endpoints here, then run from the repository root:

```bash
python -m envir generate-questions --config envir/configs/questions.yaml
python -m envir naturalize-subtasks --config envir/configs/subagent_reports.yaml
python -m envir generate-main-turns --config envir/configs/main_trajectories.yaml
```

- `questions.yaml`: Candidate construction, replay, and Question Author/Critic.
- `subagent_reports.yaml`: Oracle Macro evidence and trusted Sub-Agent reports; both passes run with one command.
- `main_trajectories.yaml`: Trusted Main-Agent trajectory Author/Critic, round budget, and journal outputs.
- Future: A separately versioned `untrusted_orchestration.yaml` when the adversarial data contract is implemented. There is **no live attack runner** in the three trusted workflows.

Paths in YAML are relative to the repository root, not the YAML file's own directory. Keep output directories different from inputs. Previously generated outputs with old code signatures should be treated as archival snapshots; start a new output directory for regenerated data.
