# Input schema

The portable runner reads JSONL. Each line contains `sample` (string),
`component` (string), `year` (integer or 0), `standards` (string), `features`
(array of strings), `label` (0/1 for source and query-pool offline references;
the runner masks query labels before selection; audit/evaluation may be -1 or
0/1 for separate offline scoring), and
`role` (`source`, `query_pool`, `audit` or `evaluation`).

The caller must create roles before running the controller. The offline JSONL
runner caches only the query-pool oracle and masks audit/evaluation labels.
The public Python trajectory function accepts already masked non-source roles.
The package checks
that samples and components do not overlap and that non-source roles are masked.
The `component` field is a user-supplied grouping identifier. This software does
not calculate ANI or SNP distances and cannot validate the meaning of a supplied
component identifier.
