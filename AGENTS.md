# Agent project workspace rules

For media tasks, follow `docs/AGENT_WORKSPACE_STANDARD.md`.

- Default to `<user material folder>/agent_work`, preserving original source names and paths.
- Initialize through `python3 -m app.agent_cli workspace init --root PATH` and pass `--workspace PATH` when submitting CLI/HTTP dubbing jobs.
- Put task configurations in `01_config`, reproducible intermediate assets in `02_work/<job_id>`, project references in `03_voices`, deliverables in `04_outputs`, execution evidence in `05_logs/<job_id>`, and disposable scripts/probes in `99_tmp/<job_id>`.
- Do not scatter task files into the Desktop root, repository root, or shared system temporary folders.
- Do not delete original materials, outputs, voice references, active task files, model weights, or service state as part of temporary-file cleanup.
- The existing backend has its own internal state/cache; do not move it live or claim workspace redirects all backend internals.
