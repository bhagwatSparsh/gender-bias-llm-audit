# Gender Bias in Large Language Models: A Comparative Counterfactual Audit of Four Models

**Authors:** Sparsh Y. Bhagwat, Tanya Singh, Isha Deshmukh

We ask four AI models to make hiring, promotion and firing decisions about **identical people, changing only the gender** (a male name, a female name, or no name; with and without he/she pronouns). We also run a separate **stereotype test** (pronouns, reference letters, role recommendations). The question: *if two applicants are identical except for gender, does the model decide differently?*

**Models:** GPT-OSS-120B, GPT-OSS-20B, Qwen-3.8-27B, Gemini-3.5-Flash-Lite.

## Main findings (reliable vs signal)

"Reliable" means the result survived correction for running many tests (Holm). "Signal" means it leans one way but did not survive.

| Finding | Strength |
|---|---|
| **One reliable decision gap, and it favours women:** GPT-OSS-120B, Indian scenarios, name + pronouns: women approved 14.1 points more often (95% CI -22.2 to -7.1, Holm p = 0.016). The other 15 approval tests are within noise. | Reliable |
| **Models link "he" to male jobs and "she" to female jobs** in ambiguous sentences: +17.1 (GPT-OSS-120B), +27.8 (GPT-OSS-20B), +27.9 (Qwen) points above chance. Gemini +27.2 but weaker (Holm p = 0.08). This shows stereotype use; it is the same for "she" as for "he". | Reliable in 3 of 4 models |
| **Gemini reacts to a name, not to gender:** in Indian scenarios any name lowers approval by about 18 points (male -18.2, female -17.2) compared with no name. | Reliable |
| **Role picks lean male** (male-stereotyped role given to male names more often): +8.1, +4.4, +6.3 points in three models; Gemini -1.9. | Signal, not reliable (smallest corrected p = 0.19) |
| **Gemini's differing decisions:** man favoured 13 times, woman 4 times (hiring 11 vs 3). Found after looking at the data. | Signal, needs a fresh confirmation test |
| **Reference letters:** no meaningful difference in agentic vs communal wording. | None |

**Bottom line:** bias appears in both directions, with a slight male tilt in the stereotype measures. We do not claim male bias is proven, and we do not claim any model is unbiased. Our design can only detect gaps of about 5-15 points; smaller effects cannot be ruled out.

A pre-set validity gate (at least half of the borderline scenarios informative for each model) was **not met by any model** (39%, 35%, 18% and 17%), so approval gaps are less sensitive than intended. This is reported in the paper.

## Study design

* **Study 2 (main decision audit):** 258 scenarios (198 borderline + 60 clear cases) across hiring, promotion and firing, each in a general and an Indian setting. Each scenario is sent in matched versions: name only, name + pronouns, no name, and a second name pair (placebo). 1,422 responses per model, 5,688 in total. Outputs per response: a decision, a suitability score (1-10) and an amount (salary, new pay or severance weeks). The gap is male minus female within the same scenario. Scenarios were selected from gender-free calibration runs.
* **Study 3 (stereotype test):** 800 prompts per model (3,200 in total): pronoun resolution, reference letters, role recommendation. Hypotheses were written down and dated 17 October 2026 before any response was collected (`docs/PREREGISTRATION_STEREO.md`).
* **Study 1 (pilot):** 48 scenarios, 1,728 responses. It led to the calibration step and the Indian/firing/placebo additions. Its raw files are not part of this repository.
* **Settings:** temperature 0.7, strict JSON output. GPT-OSS: low reasoning effort, hidden; Qwen: thinking off; Gemini: default with JSON mode.
* **Statistics:** scenario-clustered bootstrap intervals (10,000), sign-flip permutation tests, Holm correction within each outcome family, minimum detectable effect at 80% power, equivalence margins (approval +-10 points, score +-0.5, amount +-3%).

## What is in this repository

```
README.md
paper/            IEEE paper: main.tex, references.bib, figures, tables, make_figures.py,
                  and the compiled PDF (Gender_Bias_LLM_IEEE_Paper.pdf)
slides/           the presentation deck (.pptx)
code/             scenario builder, runners, parsers and analysis scripts
scenarios/        candidate scenario pools and the 258 selected scenario IDs
manifests/        every question sent to the models (Study 2, Study 3, calibration)
results/          raw model responses: results_v3.jsonl (Study 2), results_stereo_v3.jsonl (Study 3)
analysis_output/  result tables (CSV) produced by the analysis scripts
case_lists/       every pair of answers that differed between the male and female version
docs/             the dated analysis plan for Study 3
```

## Reproduce the analysis

Python 3.10+ and `pip install -r requirements.txt`. Run these from the repository root (the outputs match the committed tables exactly):

```
python code/analysis_v3.py --results results/results_v3.jsonl --manifest manifests/manifest_v3.csv --out analysis_output/study2_decisions
python code/analysis_stereo_v3.py --results results/results_stereo_v3.jsonl --manifest manifests/manifest_stereo_v3.csv --out analysis_output/study3_stereotypes
```

The first script prints a warning that the validity gate failed; that is the finding described above, not an error.

Rebuild the paper figures and tables, then the PDF (needs LaTeX):

```
cd paper
python make_figures.py
latexmk -pdf main.tex
```

## Collect new data (optional)

`code/runner_v3.py` and `code/runner_stereo_v3.py` send the manifests to the models. They need your own API keys in a file named `.env` (`GROQ_API_KEY` and `GEMINI_API_KEY`). The `.env` file is not in this repository and must never be committed. Run each script with `--help` for its options.

## Important caveats

* A single pair of differing answers is **not** proof of bias: the same prompt sent again can give a different answer (in the pronoun task, identical prompts gave non-identical answers 39%, 45%, 28% and 5% of the time for GPT-OSS-120B, GPT-OSS-20B, Qwen and Gemini).
* The case lists and the Gemini lead are post hoc and uncorrected.
* Job and role stereotype labels and the agentic/communal word lists are author-defined and not validated, especially for India.
* One run per prompt at one temperature; four models; English only; binary gender signalled by first names and pronouns.
* All scenarios and names are synthetic. No real personal data are used.

## Not included

API keys (`.env`), the raw gender-free calibration outputs (only the selected scenario IDs are included), and the Study 1 pilot raw files.
