# Wine Classification - Docker

IE 7374 - Machine Learning Operations (MLOps) | Lab 4

> **Note:** The original lab uses the Iris dataset with a Random Forest Classifier. For my implementation, I've swapped in sklearn's Wine dataset (3 classes, 13 chemical features) and replaced the Random Forest with an MLPClassifier (neural network) with StandardScaler preprocessing. The Docker containerization structure remains the same - refer to the original lab instructions for base concepts.

## Changes from Original Code

| Original (Iris) | Updated (Wine) |
|---|---|
| `load_iris()` dataset (4 features, 3 classes) | `load_wine()` dataset (13 features, 3 classes) |
| `RandomForestClassifier` | `MLPClassifier` (neural network, hidden layers: 64, 32) |
| No feature scaling | `StandardScaler` for feature normalization |
| Saves `iris_model.pkl` | Saves `wine_model.pkl` and `scaler.pkl` to `model/` directory |

## Folder Structure

```
├── src/
│   ├── main.py
│   └── requirements.txt
├── Dockerfile
└── README.md
```

## Setup & Run

**1. Clone the repo and navigate to the project directory:**

```bash
git clone <repo-url>
cd mlops-docker1-lab4
```

**2. Build the Docker image:**

```bash
docker build -t wine-classification:v1 .
```

This builds a Docker image named `wine-classification` with tag `v1` using the `Dockerfile` in the current directory. It installs `scikit-learn` and `joblib` inside the container.

**3. Run the container:**

```bash
docker run wine-classification:v1
```

This trains an MLPClassifier on the Wine dataset inside the container. You should see output like:

```
Test Accuracy: 0.9722
Model and scaler saved successfully!
```

**4. (Optional) Save the image as a tar file:**

```bash
docker save wine-classification:v1 > wine_classification.tar
# docker save -o wine_classification.tar wine-classification:v1 # for windows
```

**5. (Optional) Load a saved image:**

```bash
docker load < wine_classification.tar
# docker load -i wine_classification.tar # for windows powershell
```

## Expected Output

```
Test Accuracy: 0.9722
Model and scaler saved successfully!
```

The model trains on sklearn's Wine dataset (178 samples, 13 chemical features) and classifies wines into 3 classes (0, 1, or 2).
