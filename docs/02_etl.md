# Data Pipeline and Feature Engineering 🔧

This document explains the **ETL flow** used in the H&M Personalized Fashion Recommendation System, from raw CSV inputs to a reusable **Parquet-based feature store** that supports both training and inference.

---

## 🧭 Goal

The ETL process is designed to:
- Convert raw retail tables into **clean, consistent feature tables**
- Enable reproducible ML training through a **feature store**
- Support fast inference by keeping serving inputs **structured and ready**

---

## 📥 Inputs

The pipeline starts from three raw sources:
- **customers.csv** (user attributes)
- **articles.csv** (item catalog attributes)
- **transactions_train.csv** (user–item purchase history)

---

## ✅ Schema Validation

Before transformation, the pipeline validates:
- **Required columns** exist in each table
- **Data types** are compatible (e.g., dates, IDs)
- **Key integrity** for join fields (customer_id, article_id)

Schema validation ensures the pipeline fails early when inputs are incomplete or corrupted.

---

## 🧹 Cleaning and Standardization

Core cleaning steps include:
- Removing invalid or missing identifiers (**customer_id**, **article_id**)
- Normalizing column formats (e.g., strings, casing)
- Converting transaction dates into a standard **datetime** format
- Handling missing values for optional attributes (e.g., age)

The goal is to produce tables that are **join-safe** and consistent.

---

## 🧩 Feature Table Construction

After cleaning, the pipeline constructs canonical feature tables used by the model.

### 👤 Users Table

The users table contains one row per customer with stable profile attributes.

Typical fields:
- customer_id
- age
- club_member_status
- fashion_news_frequency

These features can help model personalization, especially when user interaction history is limited.

---

### 👗 Items Table

The items table contains one row per product (article) with descriptive attributes.

Typical fields:
- article_id
- product_type_name
- garment_group_name
- product_group_name
- color_group_name

These fields help define item identity and support meaningful retrieval behavior.

---

### 🔗 Interactions Table

The interactions table is derived from transactions and represents the implicit feedback signal.

Typical fields:
- customer_id
- article_id
- t_dat (timestamp)

This table is used to build:
- training pairs for the Two-Tower model
- evaluation splits
- aggregated summaries such as purchase counts or last purchase date

---

## 🗄️ Feature Store Output (Parquet)

All processed tables are stored as **Parquet files** for:
- faster reads vs CSV
- stable schemas
- easy reuse across phases

The feature store acts as the shared contract between:
- **FASE 2** (ETL)
- **FASE 3** (training and evaluation)
- **FASE 4** (serving and demo)

---

## 🧪 Reproducibility and Consistency

To keep outputs consistent across runs, the pipeline emphasizes:
- deterministic transformations
- explicit column selection
- consistent ID handling and joins

This helps ensure that training artifacts and demo results remain **traceable**.

---

## 🧰 How It Connects to the ML Phase

The outputs from this ETL step are directly consumed by:
- Two-Tower training (user and item inputs)
- FAISS indexing (item embedding lookup)
- Offline evaluation (time-based split logic)

Details on how these tables are consumed are documented in **docs/03_ml.md**.

---

## 📝 Notes

- Raw data is processed into feature tables to reduce coupling between phases.
- Feature store Parquet outputs are the recommended interface for both training and inference.
- This pipeline is designed to be extendable (e.g., adding new features without breaking downstream code).
