# E. coli MIC and testing-standard audit protocol

This audit is a frozen descriptive audit of the public BVBRC E. coli antimicrobial-resistance table. It does not recode labels and does not infer breakpoints. The input fields are retained as reported: drug, S/R phenotype, testing standard, standard year, measurement, unit and laboratory method.

For each genome–drug pair we count records and flag: (i) multiple distinct measurements, (ii) multiple distinct S/R labels, (iii) multiple standards or methods, and (iv) a disagreement between an available S/R label and a numeric MIC record. MIC values are parsed only into an explicit numeric value, an interval/range token, or an unparseable token; interval bounds are never converted to S/R.

The audit joins the full laboratory table to the curated S/R table by genome_id and drug when available. It also reports the fraction that can be linked to genome metadata. It leaves CLSI, EUCAST, NARMS and other standards in separate strata and reports missing years and units. No breakpoint version is reconstructed from a publication or inferred from a year.

Consequently, a successful run demonstrates data provenance, conflict rates and the evidence available for a future pre-registered harmonization study. It cannot establish cross-standard MIC equivalence, clinical validity, or blind external confirmation.
