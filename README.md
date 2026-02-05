# 🚀 ML Arena - Automated Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Smart ML Pipeline that automatically detects your problem type, preprocesses any data, and compares 8+ models to find your best fit.**

## ✨ What Makes This Special?

ML Arena takes the complexity out of machine learning by:

- 🧠 **Auto Problem Detection** - Automatically identifies if your task is regression or classification
- 🔄 **Smart Data Processing** - Handles numeric, categorical, AND text data in one pipeline
- 🎯 **Model Tournament** - Tests 8+ models and picks the winner based on generalization
- 📊 **Overfitting Diagnosis** - Tells you if your models are overfitting, underfitting, or just right
- ⚡ **Zero Config** - Just point it at your CSV and target column

## 🎭 The Problem It Solves

**Ever wonder:** *"Which ML model should I use? How do I handle text columns? Is my model overfitting?"*

**ML Arena answers:** Run one command, get all models compared with diagnostics.

---

## 🏗️ Architecture

```
📦 ML Arena
├── 🎯 models/              # 8 model implementations (Linear, SVC, KNN, etc.)
│   ├── base_model.py       # Base class for all models
│   ├── linear.py           # Linear Regression
│   ├── logistic.py         # Logistic Regression
│   ├── decision_tree.py    # Decision Tree (Classifier & Regressor)
│   ├── svc.py              # Support Vector Classifier
│   ├── svr.py              # Support Vector Regressor
│   ├── knn.py              # K-Nearest Neighbors
│   ├── gaussian.py         # Gaussian Naive Bayes
│   └── multinomial.py      # Multinomial Naive Bayes
│
├── ⚙️ processing/          # Data preprocessing pipeline
│   └── processing.py       # Auto-detects data types & transforms
│
├── 🏆 trainer_evaluator/   # Model training & evaluation
│   └── model_train.py      # Trains, compares, and diagnoses models
│
├── 📊 dataset/             # Your datasets go here
└── 🎬 main.py              # Entry point
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Ayon-coder/model_arena.git
cd model_arena

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
python main.py
```

**Then enter:**
1. Path to your CSV dataset
2. Name of your target column

### Example Run

```bash
$ python main.py
Enter your dataset path: dataset/HR_comma_sep.csv
Enter your target column: left

🎯 Problem Type Detected: Classification
📊 Compatible Models: ['logistic', 'multinomialnb', 'gaussiannb', 'svc', 'knn', 'decision_tree']

🏆 Results:
Winner: logistic
├── Train Accuracy: 0.89
├── Test Accuracy: 0.88
├── Gap: 0.01
└── Verdict: ✅ good_fit

⚠️  Other Models:
├── decision_tree: overfitting (train: 0.99, test: 0.92)
├── knn: underfitting (train: 0.54, test: 0.52)
└── svc: good_fit (train: 0.87, test: 0.86)
```

---

## 🧠 How It Works

### 1️⃣ **Problem Detection**
```python
# Automatically detects:
- Object columns → Classification
- Numeric with ≤20 unique values → Classification  
- Numeric with >20 unique values → Regression
```

### 2️⃣ **Smart Preprocessing**
```python
# Automatically handles:
- Text Columns: CountVectorizer (avg_words > 1)
- Categorical: OneHotEncoding
- Numeric: Imputation + MaxAbsScaler
```

### 3️⃣ **Model Compatibility Filtering**
```python
# Only tests models that support:
- Your problem type (regression/classification)
- Sparse matrices (if text data present)
- High-dimensional text features
```

### 4️⃣ **Winner Selection**
```python
# Picks model with smallest train-test gap
# Diagnoses each model:
- Overfitting: train >> test (gap > 0.1)
- Underfitting: both train & test < 0.6
- Good Fit: small gap, decent performance
```

---

## 📋 Supported Models

| Model | Regression | Classification | Sparse Support | Text Support |
|-------|-----------|----------------|----------------|--------------|
| **Linear Regression** | ✅ | ❌ | ✅ | ✅ |
| **Logistic Regression** | ❌ | ✅ | ✅ | ✅ |
| **SVR** | ✅ | ❌ | ✅ | ✅ |
| **SVC** | ❌ | ✅ | ✅ | ✅ |
| **Decision Tree** | ✅ | ✅ | ✅ | ❌ |
| **KNN** | ❌ | ✅ | ❌ | ❌ |
| **Gaussian NB** | ❌ | ✅ | ❌ | ❌ |
| **Multinomial NB** | ❌ | ✅ | ✅ | ✅ |

---

## 🎯 Key Features

### 🔍 Automatic Feature Detection
- Detects numeric, categorical, and **text columns** automatically
- Text detection: columns with avg words > 1 are vectorized
- Handles missing values with smart imputation

### 🏆 Model Tournament System
- Tests all compatible models for your data
- Compares train vs test accuracy
- Selects winner based on **generalization** (smallest gap)

### 📊 Overfitting Diagnosis
Every model gets diagnosed:
- **✅ Good Fit**: Small train-test gap, decent performance
- **⚠️ Overfitting**: Train accuracy >> Test accuracy (gap > 10%)
- **❌ Underfitting**: Both train and test < 60%

### ⚡ Sparse Matrix Support
- Automatically handles sparse matrices from text vectorization
- Filters out models that can't handle sparse data
- Zero memory overhead for text-heavy datasets

---

## 💡 Example Use Cases

### 📈 Predicting House Prices
```
Dataset: house_prices.csv
Target: price
→ Detects: Regression
→ Models: LinearRegression, SVR, DecisionTreeRegressor
```

### 👥 Employee Churn Prediction
```
Dataset: HR_data.csv
Target: left
→ Detects: Classification
→ Models: Logistic, SVC, DecisionTree, NaiveBayes, KNN
```

### 📧 Spam Email Detection (with text)
```
Dataset: emails.csv (with 'message' text column)
Target: is_spam
→ Detects: Classification + Text
→ Models: Logistic, MultinomialNB, SVC (filters out KNN, GaussianNB)
```

---

## 🛠️ Advanced Configuration

### Custom Model Registry
Edit `processing/processing.py` to customize model compatibility:

```python
MODEL_REGISTRY = {
    "linear": {
        "task": ["regression"],
        "sparse": True,   # Can handle sparse matrices
        "text": True      # Works with text features
    },
    # Add your own models...
}
```

### Adjust Problem Detection Threshold
```python
# In processing.py → identify_problem()
if y.nunique() <= 20:  # Change threshold here
    return "classification", y.astype(int)
```

---

## 🧪 Project Status

- ✅ Core pipeline working
- ✅ 8 models implemented
- ✅ Auto problem detection
- ✅ Text data support
- ⏳ Unit tests (coming soon)
- ⏳ Model saving/loading (coming soon)
- ⏳ Cross-validation (coming soon)
- ⏳ Hyperparameter tuning (coming soon)

---

## 🤝 Contributing

Found a bug? Want to add a model? PRs welcome!

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

MIT License - feel free to use this in your projects!

---

## 🙏 Acknowledgments

Built with:
- [scikit-learn](https://scikit-learn.org/) - ML algorithms
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [NumPy](https://numpy.org/) - Numerical computing

---

## 📧 Contact

**Ayon** - [@Ayon-coder](https://github.com/Ayon-coder)

Project Link: [https://github.com/Ayon-coder/model_arena](https://github.com/Ayon-coder/model_arena)

---

<div align="center">

### ⭐ Star this repo if you found it helpful!

**Made with ❤️ and lots of ☕**

</div>
