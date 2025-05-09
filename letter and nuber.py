import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer


# === Feature Extraction ===
def extract_hog_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features


# === Load Data ===
def load_dataset(dataset_path):
    X, y_type, y_class = [], [], []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            features = extract_hog_features(img_path)

            type_label = 'letter' if label.isalpha() else 'number'
            X.append(features)
            y_type.append(type_label)
            y_class.append(label)

    return np.array(X), np.array(y_type), np.array(y_class)


# === Plot AUROC and AUPR ===
def plot_roc_pr(y_true, y_scores, classes, title="Model"):
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_true)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    plt.figure(figsize=(12, 5))

    for i, class_name in enumerate(lb.classes_):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_scores[:, i])
        auc_score = roc_auc_score(y_bin[:, i], y_scores[:, i])
        ap_score = average_precision_score(y_bin[:, i], y_scores[:, i])

        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc_score:.2f})")
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f"{class_name} (AP={ap_score:.2f})")

    plt.subplot(1, 2, 1)
    plt.title(f"{title} - ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title(f"{title} - Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.show()


# === Train and Evaluate Multiple Models ===
def train_models(X, y, label=""):
    models = {
        'SVC': SVC(kernel='linear', probability=True),
        'RandomForest': RandomForestClassifier(),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'GradientBoosting': GradientBoostingClassifier()
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model = None
    best_acc = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\n{name} {label} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        plot_roc_pr(y_test, y_proba, np.unique(y), title=f"{name} - {label}")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    return best_model


# === Predict Single Image ===
def predict_image(image_path, type_clf, letter_clf, number_clf):
    features = extract_hog_features(image_path).reshape(1, -1)
    type_pred = type_clf.predict(features)[0]
    class_pred = letter_clf.predict(features)[0] if type_pred == 'letter' else number_clf.predict(features)[0]
    return type_pred, class_pred


# === Main ===
if __name__ == '__main__':
    dataset_path = r"C:\Users\rishi\Downloads\asl_dataset"  # Update as needed
    test_img_path = r"C:\Users\rishi\Downloads\asl_dataset\r\hand2_r_left_seg_3_cropped.jpeg"

    print("Loading dataset...")
    X, y_type, y_class = load_dataset(dataset_path)

    print("\nTraining type classifiers...")
    type_clf = train_models(X, y_type, label="Type")

    y_series = pd.Series(y_class.astype(str))
    X_letters = X[y_series.str.isalpha()]
    y_letters = y_series[y_series.str.isalpha()].values

    X_numbers = X[~y_series.str.isalpha()]
    y_numbers = y_series[~y_series.str.isalpha()].values

    print("\nTraining letter classifiers...")
    letter_clf = train_models(X_letters, y_letters, label="Letter")

    print("\nTraining number classifiers...")
    number_clf = train_models(X_numbers, y_numbers, label="Number")

    print("\nPredicting a test image...")
    t_pred, c_pred = predict_image(test_img_path, type_clf, letter_clf, number_clf)
    print(f"\nPrediction for {test_img_path}: Type = {t_pred}, Class = {c_pred}")
