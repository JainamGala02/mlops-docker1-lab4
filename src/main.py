#!/usr/bin/env python

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_data():
    """loading the Wine dataset from sklearn."""
    wine = load_wine()
    X = wine.data
    y = wine.target
    return X, y


def split_data(X, y):
    """split data into training and testing sets (80/20 split)."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    """standardize features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def fit_model(X_train, y_train):
    """train an MLPClassifier on the training data."""
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf


def save_model(clf, scaler, model_dir="model"):
    """save the trained model and scaler to disk."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, "wine_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))


if __name__ == "__main__":
    # load the Wine dataset
    X, y = load_data()

    # split into train/test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # scale features using StandardScaler
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # train an MLPClassifier on the scaled data
    clf = fit_model(X_train_scaled, y_train)

    # evaluate model accuracy on the test set
    accuracy = clf.score(X_test_scaled, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # save the trained model and scaler
    save_model(clf, scaler)
    print("Model and scaler saved successfully!")
