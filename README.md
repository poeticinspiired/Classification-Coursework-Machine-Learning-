# Phishing Website Detection: Comparative Analysis of k-NN and Naïve Bayes Algorithms

![Machine Learning](https://img.shields.io/badge/-Machine%20Learning-blue)  
![Classification](https://img.shields.io/badge/-Classification-green)  
![Python](https://img.shields.io/badge/-Python-yellow)  

A comparative analysis of k-Nearest Neighbors (k-NN) and Naïve Bayes algorithms for detecting phishing websites. Includes code, datasets, and results.  

---

## Table of Contents  
- [Repository Structure](#repository-structure)  
- [Dataset](#dataset)  
- [Methodology](#methodology)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Results](#results)  
- [Conclusion](#conclusion)  
- [License](#license)  
- [References](#references)  

---

## Repository Structure  

```plaintext
phishing-detection/
├── data/
│   ├── Training_Dataset.csv  # 8,844 samples (80% of total data)
│   └── Test_Dataset.csv  # 2,211 samples (20% of total data)
├── scripts/
│   └── knn_phishing.py  # Code for k-NN experiments
├── README.md
└── requirements.txt
```

---

## Dataset  
**Source**: [UCI Phishing Websites Dataset](https://archive.ics.uci.edu/dataset/327/phishing+websites)  
- **Training Data**: `data/Training_Dataset.csv` (8,844 samples)  
- **Test Data**: `data/Test_Dataset.csv` (2,211 samples)  
- **Features**: 30 integer/binary attributes (e.g., `url_length`, `sslfinal_state`).  
- **Target Variable**: Binary label (`1` for phishing, `0` for legitimate).  

---

## Methodology  
### 1. **Custom k-NN vs. Scikit-learn k-NN**  
- **Preprocessing**: Data normalized with `StandardScaler`.  
- **Train-Test Split**: 80% training (8,844 samples), 20% testing (2,211 samples).  
- **k Values**: 3, 5, 7.  
- **Results**: Scikit-learn’s k-NN slightly outperformed the custom implementation due to distance-weighted voting.  

### 2. **Optimizing Scikit-learn k-NN**  
- **Parameters Tested**:  
  - **k**: 3 vs. 7  
  - **Distance Metrics**: Euclidean vs. Manhattan  
  - **Voting**: Uniform vs. distance-weighted  
- **Evaluation**: 5-fold cross-validation.  

### 3. **Naïve Bayes Experiments**  
- **Algorithm**: Gaussian Naïve Bayes.  
- **Evaluation**: 5-fold cross-validation.  

---

## Installation  
1. Clone the repository:  
   ```bash  
[   git clone https://github.com/your-username/phishing-detection.git  
](https://github.com/poeticinspiired/Classification-Coursework-Machine-Learning-.git)  
![Screenshot 2025-02-22 022857](https://github.com/user-attachments/assets/e4c9c5ff-a4a3-4018-bee0-6f6fee68f9ea)
cd phishing-detection  
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage  
### Running the k-NN Model  
Execute the `knn_phishing.py` script to train and evaluate the optimal k-NN configuration:  
```bash
python scripts/knn_phishing.py  
```

### Code Snippet: Optimal k-NN Configuration  
```python
import pandas as pd  
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score  

# Load data  
train_data = pd.read_csv('data/Training_Dataset.csv')  
test_data = pd.read_csv('data/Test_Dataset.csv')  

# Split features (X) and labels (y)  
X_train = train_data.drop('target', axis=1)  
y_train = train_data['target']  
X_test = test_data.drop('target', axis=1)  
y_test = test_data['target']  

# Normalize data  
scaler = StandardScaler()  
X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test)  

# Train model  
model = KNeighborsClassifier(n_neighbors=3, metric='manhattan', weights='distance')  
model.fit(X_train, y_train)  

# Evaluate  
y_pred = model.predict(X_test)  
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")  
```

---

## Results  
### k-NN Performance  
| Configuration            | Accuracy |
|--------------------------|----------|
| k=3, Manhattan, weighted | 96.7%    |
| k=5, Manhattan, weighted | 96.2%    |
| k=7, Euclidean, uniform  | 94.8%    |

### Naïve Bayes Performance  
| Metric          | Score  |
|----------------|--------|
| Mean Accuracy  | 74.97% |
| Mean F1-Score  | 73.82% |

### Key Trends:  
- **Manhattan distance** outperformed Euclidean due to robustness to outliers.  
- **Distance-weighted voting** improved accuracy by prioritizing closer neighbors.  
- **k=3** balanced noise reduction and locality best.  

---

## Conclusion  
### **Recommended Model:**  
k-NN with Manhattan distance, **k=3**, and distance-weighted voting achieved **96.7% accuracy**, outperforming Naïve Bayes by ~22%.  

### **Why k-NN?**  
- Handles correlated features and non-linear decision boundaries.  
- Robust to feature scale variations after normalization.  

### **Avoid Naïve Bayes Due To:**  
- Violated independence assumptions.  
- Poor performance on non-Gaussian features.  

---

## License  
This project is licensed under the **Apache-2.0** license.  

---

## References  
1. **Dataset**: F, R. M., & McCluskey, T., L. (2012). *Phishing Websites*. UCI Machine Learning Repository.  
2. **Scikit-learn**: Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*.  

