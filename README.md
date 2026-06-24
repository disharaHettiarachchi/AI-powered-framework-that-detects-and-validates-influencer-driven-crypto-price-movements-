# Hybrid Sentiment-Credibility Scoring for Cryptocurrency Price Direction

Code and data for the MSc thesis "An AI-powered framework that detects and validates influencer-driven crypto price movements to make trading decisions."

This repository contains the labelling, scoring, validation, and significance-testing code behind the thesis. The work studies whether Reddit sentiment, weighted by source credibility, carries any short-horizon predictive signal for cryptocurrency price direction, and it audits the quality of the sentiment labels themselves.

## Overview

The framework combines lexicon and transformer sentiment (VADER and FinBERT) into a hybrid sentiment score, then weights that score by a source-credibility estimate. Three credibility scoring methods are compared: a raw multiplicative score, a log-multiplicative score for numerical stability, and a Bayesian (Beta-Binomial, empirical-Bayes) score that shrinks poorly-sampled sources toward a neutral prior. Predictions are evaluated for direction (up, down, neutral) at the 1 hour, 6 hour, 12 hour, and 24 hour horizons against a majority-class baseline.

The headline finding is deliberately cautious. There is a small directional edge at the 6 hour and 12 hour horizons, but under strict testing (permutation and McNemar tests with a Holm correction across horizons) it is not statistically distinguishable from chance, and a label-reliability audit shows the underlying sentiment labels are weak. The framework is therefore positioned as decision support, not a standalone trading system.

## Key results

| Check | Result |
| --- | --- |
| Label reliability (Cohen's kappa vs human gold standard) | LLM annotator 0.44, FinBERT 0.29, VADER 0.09 |
| Significance of the directional edge (Holm-corrected) | No horizon significant at 0.05 |
| Walk-forward validation, 6 hour horizon | Mean balanced accuracy 0.511 (SD 0.008), above 0.50 in each test year |

## Repository structure

```
.
├── notebooks/
│   ├── 01_regenerate_labels.ipynb          # Rebuild direction labels from price data and horizons
│   ├── 02_label_validation_LLM.ipynb       # Human gold standard, LLM/VADER/FinBERT agreement (Cohen's kappa)
│   └── 03_significance_walkforward_features.ipynb  # Permutation/McNemar tests, Holm correction, walk-forward
├── data/
│   ├── labels_regenerated.csv              # Regenerated direction labels
│   └── llm_labels_500.csv                  # LLM-annotated audit sample
├── results/
│   ├── reliability_table.csv              # Cohen's kappa per labeller
│   ├── significance_results.csv           # Balanced accuracy and p-values by horizon
│   ├── walkforward_6h.csv                 # Year-by-year walk-forward results (6 hour horizon)
│   ├── label_reliability.png             # Figure: sentiment-label agreement
│   └── walkforward_6h.png                # Figure: walk-forward balanced accuracy
├── requirements.txt
├── LICENSE
└── README.md
```

(Adjust the paths above to match how you organise the files in the repository.)

## Methodology summary

1. Labelling. Reddit posts are mapped to forward price direction at each horizon. Labels are regenerated reproducibly in notebook 01.
2. Sentiment. A hybrid score combines VADER and a FinBERT directional signal. A model-confidence component captures sentiment uncertainty.
3. Credibility. Source, content, and model credibility are combined. The Bayesian variant uses a weak symmetric Beta(2, 2) prior and updates it with each source's history of correct predictions, shrinking small-sample sources toward 0.5.
4. Evaluation. Balanced accuracy, precision, recall, F1, and ROC-AUC are reported against a majority-class baseline, using a chronological train-test split with non-leaky feature construction.
5. Validation. Label reliability is measured against a blind human gold standard with Cohen's kappa (notebook 02). The directional edge is re-tested with permutation and McNemar tests and a Holm correction, and robustness is checked with walk-forward validation (notebook 03).

## Requirements

- Python 3.10 or later
- Core libraries: pandas, numpy, scikit-learn, scipy, matplotlib, vaderSentiment, transformers, torch (for FinBERT)
- For the LLM annotation step: an OpenAI-compatible API client

Install with:

```bash
pip install -r requirements.txt
```

## How to reproduce

1. Clone the repository and install the requirements.
2. Place the source dataset where the notebooks expect it (see the path variables at the top of each notebook).
3. Run the notebooks in order:
   1. `01_regenerate_labels.ipynb` to build the labels.
   2. `02_label_validation_LLM.ipynb` to reproduce the reliability table and figure.
   3. `03_significance_walkforward_features.ipynb` to reproduce the significance and walk-forward results.

Outputs are written to the `results/` folder and match the tables and figures in the thesis.

## Data availability

The analysis uses a historical Reddit cryptocurrency dataset. Please check the original dataset's licence before redistributing it. If the licence does not permit redistribution, link to the source here and include only a small sample for testing rather than the full dataset.

## Secrets and configuration

Do not commit API keys or credentials. Keep them in a local `.env` file that is listed in `.gitignore` (for example OpenAI, Reddit, or other service keys used by the notebooks). The notebooks read keys from the environment.

## Citation

If you use this code, please cite the thesis:

> Hettiarachchi, D. W. M. M. (2026). An AI-powered framework that detects and validates influencer-driven crypto price movements to make trading decisions. MSc thesis, University of Sri Jayewardenepura.

## Author and supervision

- Author: Dishara Hettiarachchi (Registration No. MSC/DSA/138), MSc in Data Science and Artificial Intelligence, University of Sri Jayewardenepura.
- Supervisors: Prof. T. G. I. Fernando and Dr. U. Dikwatta.

## License

Released under the MIT License. See `LICENSE` for details. (Change this if your faculty or dataset terms require a different licence.)
