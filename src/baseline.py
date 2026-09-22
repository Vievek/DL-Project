"""
Baseline — TF-IDF + Logistic Regression (traditional ML, NOT one of the 4 required deep-learning
models). Useful as a reference point in the Results table to show what the deep models buy you
over a simple approach.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier


def build_baseline(max_features: int = 20000):
    """
    Fit TfidfVectorizer on TRAIN texts only (same leakage rule as everywhere else), then wrap
    LogisticRegression in MultiOutputClassifier for the 6 independent binary labels.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
    return vectorizer, classifier


# TODO: fit vectorizer on train texts, transform train/val/test, fit classifier, evaluate with the
# same metrics (macro-F1, per-class precision/recall, ROC-AUC) as the 4 deep models for a fair
# side-by-side comparison in the report.
