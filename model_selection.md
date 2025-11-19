
## Selected Models
1. **Logistic Regression (with TF-IDF embeddings)**
2. **XGBoost Classifier (with TF-IDF embeddings)**

---

# 1. Logistic Regression

### How It Works
A linear classifier that models the probability of a class using weighted TF-IDF features.

### Strengths
- Extremely fast training (seconds).
- Works very well with sparse TF-IDF vectors.
- Highly interpretable — reveals which words most influence predictions.
- Standard baseline in NLP research.
- Simple to implement and evaluate.

### Weaknesses
- Only captures linear relationships.
- Cannot learn interactions between words.
- Lower maximum accuracy than tree-based models.

### Assumptions
- Features are independent.
- Relationship between features and label is approximately linear.
- Input must be fixed-length numeric vectors.

### Do We Follow These Assumptions?
- Fixed-length numeric vectors — TF-IDF provides this.  
- LR still performs well even when linear separability is imperfect.  
- Independence assumption is violated (words co-occur), but LR is robust.

---

# 2. XGBoost

### How It Works
XGBoost uses gradient-boosted decision trees, where each tree corrects errors from previous ones.

### Strengths
- Captures non-linear patterns and feature interactions.
- Excellent performance with high-dimensional TF-IDF vectors.
- Includes regularization to prevent overfitting.
- More accurate than Random Forest on most text datasets.
- Produces feature importance scores.

### Weaknesses
- More hyperparameters to tune.
- Less interpretable than Logistic Regression.
- Slightly longer training time.

### Assumptions
- Input is fixed-length numeric vectors.
- Samples are independent.
- Nonlinear boundaries exist in the data.
- TF-IDF contains meaningful signal.

### Do We Follow These Assumptions?
- All assumptions are satisfied.  
- Nonlinear patterns exist in fake vs real news.  
- TF-IDF provides ideal inputs for boosted trees.

---

