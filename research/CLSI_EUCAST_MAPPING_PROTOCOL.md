# Versioned AST/MIC mapping protocol

EUCAST official breakpoint tables (historical and current) are archived under `D:\NCS_data\reference\eucast` with source URLs and SHA-256 hashes. The extracted target rows are in `eucast_target_breakpoint_audit.json`.

For each laboratory record, map an MIC or zone only when all required fields are present: species, drug formulation, testing method, standard, standard year/revision, indication/site where the table is indication-specific, units and censoring direction. Use the table valid on the recorded test date or explicitly recorded revision. Values in brackets, ATU, ECOFF-only rows and special notes remain separate categories unless the protocol explicitly defines their interpretation.

Records lacking a revision/year, with conflicting methods or units, or with an unresolved footnote remain `unmappable`. `<=` and `>=` values are interval-censored and are not silently assigned to S/R. CLSI mapping requires a licensed, versioned table supplied by the data owner; it is not reconstructed here.

The mapping artifact must include raw measurement, original label, mapped label, table version, rule ID, exclusion reason and code/input hashes. It must be evaluated as a new label version; old prediction results cannot be relabelled in place.
