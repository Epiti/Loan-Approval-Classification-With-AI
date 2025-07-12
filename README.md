# Loan Approval Classification with AI

A machine learning pipeline for predicting loan approval status using both traditional ML models and neural networks.

---

## 🚀 Features

- Clean, modular code structure
- Data preprocessing and feature engineering
- Multiple machine learning models (KNN, Logistic Regression, Decision Tree, Random Forest)
- Neural network implementation with TensorFlow/Keras
- Evaluation metrics and reporting
- Easy environment setup

---

## 📁 Project Structure

```
AI_PROJECT/
│
├── models/
│   ├── neural_networks.py
│   └── traditional_ml.py
│
├── utils/
│   ├── data_preprocessing.py
│   └── evaluation.py
│
├── main.py
├── README.md
├── LICENSE
├── Requiments.txt
├── .gitignore
└── loan_data1.csv  # (not tracked in git, see .gitignore)
```

---

## 🛠️ Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Epiti/Loan-Approval-Classification-With-AI.git
   cd Loan-Approval-Classification-With-AI
   ```

2. **Create and activate a virtual environment (recommended):**
   ```sh
   python -m venv env_ai
   # On Windows:
   env_ai\Scripts\activate
   # On Mac/Linux:
   source env_ai/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r Requiments.txt
   ```

---

## 📊 Usage

1. **Add your dataset :**  
   Place `loan_data1.csv` in the project root (not tracked by git).

2. **Run the main script:**
   ```sh
   python main.py
   ```

---

## 📦 Dataset

- [Loan Approval Classification Dataset on Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data?resource=download#)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Kaggle Dataset Author](https://www.kaggle.com/taweilo)
- [Scikit-learn](https://scikit-learn.org/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
