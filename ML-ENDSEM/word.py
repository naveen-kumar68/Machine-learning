import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

DATASET_PATH = "path_to_dataset"
VIDEOS_DIR = os.path.join(DATASET_PATH, 'videos')
JSON_FILE_PATH = os.path.join(DATASET_PATH, 'WLASL_v0.3.json')

def load_video_class_mapping(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    video_class_map = {
        instance['video_id']: entry['gloss']
        for entry in data for instance in entry['instances']
    }
    return video_class_map

def extract_video_features(video_path, max_frames=20, resize_dim=(64, 64)):
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = 0
    prev_frame = None
    color_hist_size = 512
    motion_feat_size = 2

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize_dim)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        hist = cv2.normalize(hist, hist).flatten()
        motion_feat = np.zeros(motion_feat_size)
        if prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_feat = [np.mean(mag), np.std(mag)]
        frame_features = np.concatenate([hist, motion_feat])
        features.append(frame_features)
        prev_frame = gray
        frame_count += 1

    cap.release()

    if frame_count < max_frames:
        padding = np.zeros((max_frames - frame_count, color_hist_size + motion_feat_size))
        features.extend(padding)

    features = np.array(features)[:max_frames]
    return features.flatten()

def prepare_data(videos_dir, video_class_map, max_samples_per_class=None, max_videos=2000):
    features, labels, paths, class_counts = [], [], [], {}
    video_files = glob(os.path.join(videos_dir, '*.mp4'))[:max_videos]

    for i, vf in enumerate(video_files):
        vid = os.path.basename(vf).split('.')[0]
        if vid not in video_class_map:
            continue
        cls = video_class_map[vid]
        if max_samples_per_class and class_counts.get(cls, 0) >= max_samples_per_class:
            continue
        try:
            feats = extract_video_features(vf)
            features.append(feats)
            labels.append(cls)
            paths.append(vf)
            class_counts[cls] = class_counts.get(cls, 0) + 1
        except Exception as e:
            print(f"Error processing {vf}: {e}")

    feature_length = len(features[0]) if features else 0
    for i, feat in enumerate(features):
        if len(feat) != feature_length:
            features[i] = np.pad(feat, (0, feature_length - len(feat)), mode='constant')

    return np.array(features), np.array(labels), paths

def evaluate_models(X_train, y_train, X_test, y_test, label_encoder):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM": SVC(kernel='linear', probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    ensemble = VotingClassifier(
        estimators=[('rf', models["Random Forest"]), ('svc', models["SVM"]), ('lr', models["Logistic Regression"])],
        voting='soft'
    )
    models["Ensemble"] = ensemble

    results = {}
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv)
            print(f"Cross-validation scores for {name}: {scores}")
        except ValueError as e:
            print(f"CV failed for {name}: {e}")
            scores = []
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=label_encoder.inverse_transform(np.unique(y_test)))
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=label_encoder.inverse_transform(np.unique(y_test))).plot(
            cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix: {name}")
        plt.show()

        results[name] = {
            "Accuracy": acc,
            "Classification Report": report,
            "Predictions": list(zip(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred))),
            "Cross Val Score": scores
        }
        print(f"\n{name} Accuracy: {acc * 100:.2f}%\n")
        print(report)

    return results

if __name__ == '__main__':
    video_class_map = load_video_class_mapping(JSON_FILE_PATH)
    X, y, video_paths = prepare_data(VIDEOS_DIR, video_class_map)
    if len(X) < 100:
        raise ValueError("Not enough videos (need at least 100)")

    X_train, y_train = X[:50], y[:50]
    label_encoder = LabelEncoder()
    y_train_encoded = label
