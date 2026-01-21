# End-to-End System Storytelling 📘

This document presents the **complete narrative** of the H&M Personalized Fashion Recommendation System, explaining the motivation, design decisions, and how all components connect from data to demo.

---

## 🎯 Problem Context

Fashion retail platforms operate with:
- extremely large product catalogs
- sparse and implicit user interaction data
- strong expectations for fast and relevant recommendations

A practical recommendation system must balance **personalization, scalability, and simplicity**, while remaining interpretable and reproducible.

---

## 🧭 Design Philosophy

The system is designed with the following principles:
- **retrieval-first architecture** for scalability
- clear separation between data, model, and serving layers
- reproducibility over ad-hoc experimentation
- realistic trade-offs aligned with production systems

Rather than maximizing model complexity, the focus is on **end-to-end system clarity**.

---

## 🗂️ Data to Features

The pipeline begins with raw H&M retail data:
- customers
- articles
- transactions

These sources are transformed through a structured ETL process into:
- user features
- item features
- interaction tables

All outputs are stored as **Parquet-based feature tables**, forming a reusable feature store shared across training and inference.

---

## 🧠 Model and Retrieval

The core recommendation engine is a **Two-Tower retrieval model**.

Key characteristics:
- users and items are encoded independently
- embeddings are learned from implicit purchase behavior
- relevance is measured through vector similarity

After training, item embeddings are indexed using **FAISS**, enabling fast top-K retrieval even at scale.

---

## 📊 Evaluation and Analysis

Model performance is evaluated offline using:
- Recall@K
- NDCG@K

Evaluation follows a **time-aware split** to better reflect real-world usage.

Results are interpreted through:
- comparison with a popularity-based baseline
- qualitative analysis of personalization behavior

Detailed metrics and interpretations are documented in the **ml/reports** directory.

---

## 🔌 Serving and Demo

The trained model is served using a **FastAPI backend**.

The serving layer:
- loads trained artifacts and FAISS index at startup
- exposes clean inference endpoints
- applies a cold-start fallback strategy when needed

A lightweight **HTML-based demo UI** allows users to explore:
- customer profiles
- recent purchase history
- personalized recommendation outputs

This setup enables full end-to-end validation in a local environment.

---

## ⚖️ Trade-Offs and Limitations

Several deliberate trade-offs were made:
- offline evaluation instead of online experimentation
- static embeddings instead of real-time updates
- single-stage retrieval instead of multi-stage ranking

These choices prioritize **system transparency and robustness** over maximum performance.

---

## 🔮 Future Direction

The system provides a strong foundation for future extensions, including:
- multi-stage recommendation pipelines
- online feedback integration
- richer feature representations
- public-facing deployment

These improvements can be built incrementally without redesigning the core system.

---

## 🏁 Closing Summary

This project demonstrates how a **modern recommendation system** can be built as a coherent, end-to-end pipeline:
- grounded in real data
- supported by clear engineering decisions
- designed with practical constraints in mind

The result is a **portfolio-grade system** that emphasizes understanding, structure, and real-world applicability over isolated model performance.
