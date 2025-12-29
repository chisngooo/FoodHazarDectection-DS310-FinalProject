# Food Hazard Detection with Multi-task Learning and Focal Loss

This repository presents our system for **food hazard detection from web-based reports**, developed for **SemEval-2025 Task 9**.  
The task aims to jointly predict **hazard-category** and **product-category** from food safety recall texts collected from public web sources.

---

## 🧠 Method Overview

Our approach is built upon **Transformer-based language models** and focuses on achieving high performance while maintaining a **lightweight and reproducible pipeline**.

Key components include:

- **Robust text preprocessing**: HTML cleaning, boilerplate removal, sentence deduplication, and domain-specific entity normalization.
- **Token-level text chunking** with overlap to handle long documents under Transformer input limits.
- **Multi-task learning** to jointly model hazard-category and product-category, leveraging their semantic correlation.
- **Focal Loss** to address severe class imbalance and long-tail label distribution.
- **Comprehensive data augmentation**, targeting not only rare labels but also semantically overlapping and easily confused classes.
- **Two-level ensemble strategy**:
  - Chunk-level aggregation (mean pooling).
  - Model-level soft voting between RoBERTa and DeBERTa.

---

## 📊 Dataset

- Provided by **SemEval-2025 Task 9**
- 5,984 samples collected from food safety authorities worldwide
- Highly imbalanced label distributions (long-tail)
- Multilingual, dominated by English and German

---

## 🏆 Results

- **Evaluation metric**: Macro-F1 (official SemEval metric)
- **Best result**: **0.8042 macro-F1** using **TITLE + TEXT**
- **Ranking**: Top-tier performance (Top 3–5 range)

| Model / System | Macro-F1 |
|---------------|----------|
| RoBERTa (single) | 0.8027 |
| DeBERTa (single) | 0.7349 |
| Ensemble (best weight) | **0.8042** |

RoBERTa alone already shows strong multilingual generalization, while DeBERTa provides complementary gains in ensemble.

---

## ⚠️ Notes on Grid Search

We observe that **grid search on the public test set leads to unstable weights** and degraded performance (0.7842).  
Selecting ensemble weights based on the **validation set** yields more reliable and generalizable results.

---

## 🔮 Future Work

- Incorporating temporal and geographical metadata
- Exploring specialized multilingual encoders
- Integrating domain knowledge or external food safety ontologies

---

## 📌 References

- SemEval-2025 Task 9  
- RoBERTa, DeBERTa  
- Focal Loss  
- Multi-task Learning

---

**Authors**: Ngô Minh Trí, Nguyễn Đình Khôi  
**Year**: 2025
