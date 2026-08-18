# Customer Churn Prediction

A machine-learning project for predicting whether a bank customer is likely to leave the bank. The notebook compares classic classification models with a neural network, then evaluates and visualizes their performance.

## Project overview

Customer retention is a key business problem. This project uses customer account and demographic information to predict the `Exited` target:

- `0` — the customer remains with the bank
- `1` — the customer leaves the bank

## Dataset

The project uses [`Churn_Modelling.csv`](./Churn_Modelling.csv), containing 10,000 customer records and these main fields:

| Category | Features |
| --- | --- |
| Customer profile | `CreditScore`, `Geography`, `Gender`, `Age` |
| Account details | `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary` |
| Target | `Exited` |

The identifier columns `RowNumber`, `CustomerId`, and `Surname` are removed before modelling.

## Models

The notebook implements and compares:

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Artificial Neural Network (ANN)

The ANN uses two dense ReLU layers, a sigmoid output layer for binary classification, Adam optimization, and early stopping based on validation loss.

## Workflow

```text
Load data
  → remove identifier columns
  → one-hot encode Geography and Gender
  → split into training and test sets (80/20)
  → standardize features
  → train ML models and ANN
  → compare accuracy and review model results
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/nagwanprashant22-alt/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow jupyter
```

### 3. Launch the notebook

```bash
jupyter notebook customer_churn_rate_prediction.ipynb
```

For a local run, change the data-loading cell to:

```python
df = pd.read_csv("Churn_Modelling.csv")
```

Before feature scaling, ensure this line is present:

```python
scaler = StandardScaler()
```

## Results

The notebook creates an accuracy comparison for all four models, prints a detailed classification report for Random Forest, and plots ANN training/validation loss and accuracy over time.

## Repository structure

```text
.
├── Churn_Modelling.csv                  # Customer churn dataset
├── customer_churn_rate_prediction.ipynb # Analysis and model training
└── README.md
```

## Technologies

- Python
- Pandas and NumPy
- scikit-learn
- TensorFlow / Keras
- Matplotlib

## Notes

This project is intended for learning and experimentation. Accuracy alone may not be sufficient for a churn-use case; consider precision, recall, ROC-AUC, class imbalance, and business costs when selecting a production model.
