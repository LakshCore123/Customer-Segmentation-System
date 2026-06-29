# 🛍️ Customer Segmentation System

A production-ready **Customer Segmentation System** built using **Python**, **Machine Learning**, **RFM Analysis**, and **K-Means Clustering**. This project segments e-commerce customers based on their purchasing behavior and provides actionable business insights for targeted marketing.

---

## 📌 Project Overview

Customer segmentation helps businesses understand customer behavior by grouping customers with similar purchasing patterns. This project uses the **Online Retail Dataset** and applies **Recency, Frequency, Monetary (RFM)** analysis followed by **K-Means Clustering** to identify meaningful customer segments.

---

## 🚀 Features

- Data Loading & Cleaning
- Missing Value Analysis
- Exploratory Data Analysis (EDA)
- RFM Feature Engineering
- Feature Scaling
- K-Means Clustering
- Elbow Method
- Silhouette Analysis
- PCA Visualization
- Customer Profiling
- Marketing Strategy Generation
- Business Report Generation

---

## 📊 Dataset

**Dataset:** Online Retail Dataset

- 541,909 Transactions
- 4,338 Customers (after preprocessing)
- Source: UCI Machine Learning Repository

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- OpenPyXL

---

## 📂 Project Structure

```
Customer-Segmentation-System/
│
├── customer_segmentation.py
├── README.md
├── requirements.txt
├── .gitignore
└── online_retail.csv (Download separately)
```

---

## ⚙️ Machine Learning Workflow

```
Data Collection
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
RFM Feature Engineering
        │
        ▼
Feature Scaling
        │
        ▼
K-Means Clustering
        │
        ▼
Elbow Method
        │
        ▼
Silhouette Analysis
        │
        ▼
PCA Visualization
        │
        ▼
Customer Segmentation
        │
        ▼
Business Recommendations
```

---

## 📈 Results

- Customers Segmented: **4,338**
- Optimal Number of Clusters: **3**
- Silhouette Score: **0.4323**

### Customer Segments

| Segment | Customers |
|----------|----------:|
| Loyal Customers | 1,570 |
| New Customers | 1,790 |
| Lost Customers | 978 |

---

## 📊 Generated Outputs

The project automatically generates:

- Missing Values Heatmap
- EDA Distribution Plots
- Boxplots
- RFM Distribution
- Correlation Heatmap
- Elbow Curve
- Silhouette Analysis
- PCA Visualization
- Customer Segment Distribution
- Marketing Strategy Report
- Final Business Report

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Customer-Segmentation-System.git
```

Go to the project folder

```bash
cd Customer-Segmentation-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download the **Online Retail Dataset** and place `online_retail.csv` inside the project folder.

Run the project

```bash
python customer_segmentation.py
```

---

## 📥 Dataset Download

Download the **Online Retail Dataset** from:

- https://archive.ics.uci.edu/dataset/352/online+retail

Save the file as:

```
online_retail.csv
```

and place it in the project directory.

---

## 📌 Future Improvements

- DBSCAN & Hierarchical Clustering
- Interactive Streamlit Dashboard
- Flask/FastAPI Deployment
- Model Persistence using Joblib
- Automated Customer Recommendation Engine
- Cloud Deployment

---

## 👨‍💻 Author

**Lakshya Mittal**

- GitHub: https://github.com/LakshCore123
---

⭐ If you found this project useful, consider giving it a star!
