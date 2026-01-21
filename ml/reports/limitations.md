# System Limitations ⚠️

This document outlines the **key limitations** of the H&M Personalized Fashion Recommendation System, focusing on design trade-offs, technical constraints, and areas that require further improvement.

---

## 📊 Evaluation Scope

The system is evaluated using **offline metrics only**.

Limitations include:
- no real-time user feedback
- no online A/B testing
- performance measured solely on historical data

As a result, model effectiveness in a live environment cannot be fully validated.

---

## ❄️ Cold-Start Users

Personalization depends heavily on **historical interactions**.

For users with little or no purchase history:
- recommendations rely on popularity-based fallback
- individual taste cannot be inferred accurately

This limits personalization quality for new or inactive users.

---

## 🧠 Static Embeddings

User and item embeddings are:
- trained offline
- fixed during serving

This means:
- recent user behavior is not immediately reflected
- model updates require retraining and redeployment

The system does not support real-time embedding updates.

---

## 🧩 Single-Stage Retrieval

The recommender uses a **single-stage retrieval approach**.

Limitations of this design:
- no secondary ranking model
- limited use of contextual or session-level signals
- ranking quality depends entirely on embedding similarity

A multi-stage architecture could improve precision.

---

## 🧪 Feature Limitations

Feature engineering is intentionally conservative:
- limited use of temporal aggregation
- no sequential modeling
- no content-based text or image embeddings

This restricts the model’s ability to capture complex user preferences.

---

## ⚙️ Serving Constraints

The serving setup prioritizes simplicity and clarity.

As a result:
- scaling strategies are not implemented
- latency optimization is not benchmarked
- monitoring and alerting are minimal

These aspects would need enhancement for production-scale deployment.

---

## 📝 Summary

The current system demonstrates a **complete and functional recommendation pipeline**, but it is not optimized for large-scale production use.

The identified limitations provide a clear roadmap for future improvements, which are discussed in **future_improvements.md**.
