# Baseline vs Model Comparison 📊

This report compares the performance of a **baseline recommendation approach** and the **Two-Tower retrieval model** used in the H&M Personalized Fashion Recommendation System.

The goal is to understand whether the learned model provides **meaningful improvements** over simple heuristics.

---

## 🧪 Baseline Definition

The baseline approach is based on **item popularity**, where products are ranked by historical purchase frequency.

Characteristics of the baseline:
- no personalization
- same recommendation list for all users
- easy to compute and deploy

Despite its simplicity, popularity-based recommendation serves as a **strong reference point** in retail scenarios.

---

## 🧠 Model Definition

The proposed model is a **Two-Tower retrieval model** trained on historical user–item interactions.

Key characteristics:
- personalized recommendations per user
- learned user and item embeddings
- FAISS-based nearest neighbor retrieval
- ability to generalize beyond frequent items

---

## 📏 Evaluation Setup

Both approaches are evaluated using the same setup:
- time-aware data split
- held-out interactions per user
- identical candidate item space
- top-K retrieval comparison

This ensures that performance differences are **attributable to the model**, not the evaluation process.

---

## 📈 Evaluation Metrics

The comparison focuses on ranking-based metrics:
- **Recall@K**
- **NDCG@K**

These metrics measure how well each approach retrieves relevant items and how well it ranks them.

---

## 🆚 Comparative Results

Overall observations from the evaluation:
- the Two-Tower model consistently outperforms the popularity baseline
- improvements are more pronounced for active users with richer histories
- the baseline remains competitive for cold-start scenarios

The results indicate that the model captures **user-specific preferences** that the baseline cannot.

---

## 🧠 Interpretation

The popularity baseline favors frequently purchased items and ignores individual taste.

In contrast, the Two-Tower model:
- learns latent representations of users and items
- retrieves items aligned with historical behavior
- balances popularity with personalization

This explains the observed gains in both recall and ranking quality.

---

## ⚖️ Trade-Off Analysis

While the model improves recommendation quality, it introduces:
- additional training complexity
- dependency on feature pipelines
- higher serving complexity compared to the baseline

These trade-offs are acceptable given the **quality gains and scalability benefits**.

---

## 📝 Summary

The comparison demonstrates that:
- a simple baseline provides a useful benchmark
- the Two-Tower model delivers **clear personalization benefits**
- retrieval-based modeling is a suitable choice for large-scale fashion catalogs

Further analysis and limitations are discussed in the accompanying reports.
