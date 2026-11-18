# Working on KVServe

Run `python -m unittest discover -s tests -v` before submitting a change. Numerical or cache/scheduler changes need output-equivalence and ownership checks. API changes need request-validation and lifecycle checks.

Reference evidence belongs to its recorded source hashes. If runtime code changes, rerun the benchmark into a new directory and report the new environment and configuration. Do not change a timing value by editing a table without updating its supporting evidence. The source-identity audit intentionally fails when runtime source changes without corresponding measured artifacts.

For an experiment, state the question, changed factor, controls, arrival model, timing boundary and output-equivalence condition. Report regressions alongside improvements. Separate in-process timing, client-observed timing, assigned capacity and resident memory.

The original Deep-ML exercise remains in `archive/` for provenance. Maintained interfaces are under `kvserve/`. Use ordinary technical diagrams and data-derived figures; regenerate the plots using `scripts/report.py`.
