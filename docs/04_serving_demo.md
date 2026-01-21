# Serving and Demo Setup 🧪

This document describes how the H&M Personalized Fashion Recommendation System is **served and demonstrated locally**, covering the API design, inference flow, and demo interface.

---

## 🎯 Purpose

The serving layer is designed to:
- expose the trained recommender model via a **clean API**
- enable fast inference using precomputed embeddings and FAISS
- support a simple **demo experience** for exploration and validation

The focus is on **clarity and correctness**, not large-scale production optimization.

---

## 🧩 Serving Architecture

The serving stack consists of:
- a **FastAPI** backend for inference
- a local **model registry** containing trained artifacts
- a lightweight **HTML-based demo UI**

The backend loads model artifacts at startup and serves requests synchronously.

---

## 🔌 API Design

The FastAPI service exposes two primary endpoints.

### GET /health
- checks service availability
- verifies model and index are loaded correctly

This endpoint is used for basic health checks during development.

---

### POST /recommend
- accepts a customer identifier and a requested number of items
- returns a ranked list of recommended products

Inputs:
- customer_id
- top_k

Outputs:
- recommended item identifiers
- similarity scores
- cold-start indicator when fallback is applied

---

## 🔁 Inference Flow

The recommendation flow follows these steps:

1. Receive request with customer_id  
2. Check user history availability  
3. Encode user representation or trigger cold-start fallback  
4. Retrieve top-K items using **FAISS nearest neighbor search**  
5. Return ranked results to the client  

This flow ensures **low-latency retrieval** even with a large item catalog.

---

## ❄️ Cold-Start Handling

If a user has insufficient interaction history:
- personalization is not reliable
- the system falls back to **popular items**

Popularity is computed from historical transaction frequency and stored as a lightweight lookup table.

This guarantees that the API always returns valid recommendations.

---

## 🖥️ Demo Interface

The demo interface is implemented using **HTML templates** and rendered by the FastAPI backend.

The demo allows users to:
- input or select a customer identifier
- view basic customer profile information
- inspect recent purchase history
- request and visualize recommendations

The UI is intentionally minimal and marketplace-like to keep the focus on system behavior.

---

## ▶️ Local Execution

A typical local workflow includes:
- starting the FastAPI server
- accessing the demo via a browser
- sending test requests to the API endpoints

This setup allows end-to-end validation without requiring external infrastructure.

---

## 🧠 Design Considerations

Several design decisions guide the serving layer:
- separation between training and inference
- loading artifacts once at startup
- avoiding unnecessary dependencies in the demo layer

These choices simplify debugging and improve system transparency.

---

## 📝 Notes

This document focuses on **local serving and demonstration**.  
Deployment strategies, scalability concerns, and public access considerations are discussed separately in **docs/05_storytelling.md**.
