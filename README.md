# PAI Dashboard — Platform Asymmetry Index

Interactive Streamlit dashboard for exploring **Platform Asymmetry Index (PAI)** research results.

## What is PAI?

PAI measures power asymmetry in social media Terms of Service and Privacy Policies across 11 major platforms (Meta, YouTube, WhatsApp, Reddit, LinkedIn, TikTok, Twitter/X, Instagram, Pinterest, Discord, Google).

**Four Dimensions:**
- **Complexity** — Syntactic complexity (avg sentence length)
- **Agency Asymmetry** — Platform vs user pronoun ratio
- **Formality** — Formal legal term density
- **Discretion** — Platform power language ratio

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| Rankings | PCA-weighted platform rankings |
| Dimensions | 4-dimensional radar chart |
| Agency Asymmetry | Platform vs user visibility analysis |
| Discretion | Discretion language deep-dive |
| Detailed Metrics | Full metrics table with CSV export |
| Time Series | Policy evolution 2006-2025 with regulatory event markers |
| Regulatory Evasion | Pre-GDPR vs post-CCPA discretion language changes |
| Privacy vs ToS | Document type separation analysis |
| Causal Analysis | Difference-in-Differences (DiD) results |
| Validation | Bootstrap, permutation, normalization sensitivity, ITS |

## Data

- **1,752 policy documents** from 11 Tier 1 social media platforms
- **Temporal coverage**: 2005-2025 (up to 21 years for Meta)
- **Source**: TransparencyDB

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Research

Target journal: *Regulation & Governance*

---

*Research use only.*
