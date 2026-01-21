# Data Understanding 🧾

This document describes the **dataset used in the H&M Personalized Fashion Recommendation System**, including its structure, characteristics, and how it is prepared for downstream processing.

---

## 📦 Dataset Source

The project uses the **H&M Personalized Fashion Recommendations dataset**, which contains historical transaction data from a large-scale fashion retailer.

The dataset represents **real-world retail behavior**, making it suitable for building and evaluating personalized recommendation systems.

---

## 🗂️ Raw Data Files

The raw dataset consists of three main CSV files:

1. **customers.csv**  
   Contains demographic and membership information for each customer.

2. **articles.csv**  
   Contains product-level information, including category, garment type, and descriptive attributes.

3. **transactions_train.csv**  
   Contains historical purchase records linking customers and articles over time.

Each file serves a distinct role and is required to construct a complete user–item interaction graph.

---

## 👤 Customers Data

The **customers table** provides user-level attributes that are useful for personalization.

Key attributes include:
- customer_id
- age
- club_member_status
- fashion_news_frequency

These features help represent **user preferences and behavioral context**, especially for users with limited interaction history.

---

## 👗 Articles Data

The **articles table** represents the product catalog.

Key attributes include:
- article_id
- product_type_name
- garment_group_name
- product_group_name
- color_group_name

These attributes are used to construct **item representations** and support similarity-based retrieval.

---

## 🧾 Transactions Data

The **transactions table** records historical purchase events.

Key attributes include:
- customer_id
- article_id
- t_dat (transaction date)

This table forms the **core interaction signal** used for training the recommendation model, representing implicit feedback from users.

---

## 🔄 Data Characteristics

Several important characteristics of the dataset influence system design:

- **Implicit feedback only** (no ratings or explicit preferences)
- **High sparsity**, as most users interact with only a small fraction of items
- **Temporal nature**, where interaction time matters for training and evaluation

These characteristics motivate the use of **retrieval-based models** instead of traditional rating prediction approaches.

---

## 🧹 Initial Data Preparation

Before feature engineering, the raw data undergoes basic preparation steps:

- Removal of invalid or missing identifiers
- Standardization of column names
- Type casting for dates and categorical fields

No heavy aggregation is performed at this stage to preserve **granularity for downstream processing**.

---

## 🎯 Role in the Overall System

The raw dataset serves as the **foundation for all subsequent phases**, including:

- Feature engineering and feature store creation
- Model training and evaluation
- Offline analysis and reporting

A clear understanding of the data ensures that **model behavior and limitations** can be interpreted correctly.

---

## 📝 Notes

The raw CSV files are **not committed to the repository** due to size considerations.  
All downstream processes operate on **processed Parquet files** generated during the data pipeline phase.

Further details on data transformation and feature generation are documented in **docs/02_etl.md**.
