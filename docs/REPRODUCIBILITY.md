# Reproducibility checklist

1. Record Python version, operating system and package versions.
2. Freeze the JSONL input hash and the exact role-assignment protocol.
3. Save the command, seed, query budgets and batch size.
4. Keep source/query/audit/evaluation samples and components disjoint.
5. Mask audit and evaluation labels before selection; retain a separate offline
   scoring copy that is never passed to the controller.
6. Preserve unavailable configurations and zero/small resistant denominators.
7. Write a manifest with input, code and output SHA-256 values.
8. Re-run tests after each code change. A code change affecting a formal method
   requires a new versioned result directory and a new validation.

The `historical_frozen` directory is intentionally separate from the portable
implementation. Its files retain hardcoded paths so that their hashes remain
identical to the reported study source. Do not edit them in place.
