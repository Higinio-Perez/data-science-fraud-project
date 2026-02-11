# Fraud Detection Project – End-to-End Machine Learning Pipeline

This project implements a complete, realistic fraud detection pipeline, from synthetic data generation of transactions to model deployment decisions and interpretability.

The objective is to simulate how fraud detection systems are built in production environments and how business constraints shape modeling decisions.

---

## Phase 1 – Dataset & EDA

Goal: understand the problem and the data.

- Generated realistic synthetic fraud data.

- Explored:
  - class imbalance,
  - transaction amount distributions,
  - night vs day behavior,
  - fraud rate differences.
- Identified strong predictive signals:
  - amount,
  - night transactions,
  - cross-border activity,
  - device risk,
  - transaction velocity.

Outcome:
A clear understanding of the fraud detection problem and its class imbalance nature.

---

## Phase 2 – Baseline Model (Logistic Regression)

Goal: build a simple, interpretable baseline.

- Trained Logistic Regression with:
  - feature scaling,
  - class weighting to handle imbalance.
- Evaluated with:
  - ROC-AUC,
  - PR-AUC,
  - precision, recall, confusion matrix.

Outcome:
A solid baseline model with strong discriminative power and interpretability.

---

## Phase 3 – Tree-Based Models

Goal: test more powerful non-linear models.

Models tested:
- Random Forest
- Gradient Boosting

Findings:
- Tree models provided strong ROC-AUC.
- Performance gains over Logistic Regression were marginal.
- Tree models were less stable under threshold tuning.
- Interpretability was lower compared to Logistic Regression.

Outcome:
Tree models were competitive but did not justify replacing Logistic Regression in this context.

---

## Phase 4 – Threshold Optimization (Business-Aware Decision Making)

Goal: convert model probabilities into operational decisions.

Two threshold strategies:
- Minimum precision constraint (e.g., ≥ 20%)
- Cost-based optimization:
  - FP cost = 1
  - FN cost = 10

Final decision:
- Selected cost-optimized threshold ≈ **0.906**
- This threshold:
  - strongly reduces false positives,
  - accepts that some frauds will be missed.

This reflects real-world trade-offs:
fraud systems prioritize alert quality over catching every fraud.

---

## Phase 5 – Interpretability & Final Model Selection

Final model:
- Logistic Regression

Reasons:
- Comparable performance to Gradient Boosting.
- Better stability under threshold optimization.
- Interpretability of coefficients.
- Easier to explain to business stakeholders and regulators.

Local interpretability:
- Individual predictions were inspected.
- High-risk transactions align with domain intuition:
  - high amount,
  - risky devices,
  - high velocity,
  - night-time activity,
  - cross-border transactions.

---

## Final Conclusions

This project demonstrates:

- How model evaluation must go beyond accuracy.
- Why ROC-AUC alone is insufficient in imbalanced problems.
- How threshold selection is a business decision, not a technical default.
- Why simpler models are often preferred in production systems.
- How fraud detection systems are intentionally conservative.

This pipeline mirrors real-world fraud detection workflows used in fintech and banking environments.

---

## How to Run

1. Create environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run notebooks in order
- 01_eda.ipynb
- 02_baseline_logistic.ipynb
- 03_tree_models.ipynb
- 04_threshold_optimization.ipynb
- 05_interpretability.ipynb

---

## Author Notes

This project was designed as a portfolio-ready, interview-grade case study, focusing on:

- Realistic modeling decisions,
- Business trade-offs,
- Interpretability,
- Deployment thinking.