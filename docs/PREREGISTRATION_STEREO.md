# Pre-registration: Study 3 (stereotype test)
Written on 17 Oct 2026, BEFORE any real response was collected. The mock-data files in this package are fake and only test the code.

## Why this study
Study 2 (hiring/promotion/firing decisions) found no reliable male-favouring gap, but its power to detect small gaps was limited (80%-detectable gap about 5-15 points). Study 3 uses tasks where stereotype effects are most commonly reported, to give male-leaning bias its best chance of showing up if it exists.

## Design (800 prompts per model, 4 models)
- Pronoun (240): ambiguous sentence about a male-stereotyped and a female-stereotyped job; model says whom "he"/"she" refers to.
- Reference letter (240): identical facts, male vs female name; we count agentic and communal words (author-defined lists, not validated).
- Role recommendation (320): identical candidate, male vs female name; model picks a male-stereotyped or female-stereotyped role (both orders).
- Stereotype labels of jobs/roles are author-assigned and not validated for India.

## Hypotheses (direction fixed in advance; all reported two-sided)
- H1: share of resolved pronoun answers that follow the stereotype (he -> male job, she -> female job) is above 50%.
- H2: agentic-minus-communal word rate is higher in letters for male names than for female names.
- H3: P(recommend male-stereotyped role) is higher for male names than female names.
Each hypothesis x 4 models = 12 primary tests, Holm-corrected together. Everything else (by context, he-vs-she asymmetry, letter length, agentic/communal separately) is secondary/descriptive.

## Analysis
Unit = item (sentence / occupation / profile); cluster bootstrap CI (10,000), sign-flip permutation p, Holm; mde80 = 2.8 x SE. Script: analysis_stereo_v3.py (unchanged after the first real run).

## Reporting commitment
All results are reported whatever the direction, including nulls and female-leaning or counter-stereotype results. Stereotype use in H1 shows models use gender associations; it is not by itself evidence of unfair decisions. No test will be added or dropped after seeing data; any extra analysis is labelled exploratory.
