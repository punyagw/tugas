
# -------------------------
# 1) IMPORT LIBRARY
# -------------------------
# Di bagian ini kita mengimpor pustaka yang diperlukan.
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt

# -------------------------
# 2) USER CONFIG / PATH
# -------------------------
# Ubah PATH_DATA ke lokasi dataset Anda (CSV) serta kolom target ('target_col').
PATH_DATA = "data.csv"      # <-- ganti dengan path file CSV Anda
TARGET_COL = "target"       # <-- ganti nama kolom target Anda

# -------------------------
# 3) LOAD DATA
# -------------------------
# Kita baca CSV ke DataFrame. Jika sudah punya DataFrame di memori,
# Anda bisa melewati bagian ini.
if not os.path.exists(PATH_DATA):
    # Jika file tidak ditemukan, buat contoh data sintetis agar script tetap runnable.
    from sklearn.datasets import make_classification
    X_synth, y_synth = make_classification(n_samples=1000, n_features=20, n_informative=8,
                                           n_redundant=2, n_clusters_per_class=2, weights=[0.8,0.2],
                                           random_state=42)
    df = pd.DataFrame(X_synth, columns=[f"f{i}" for i in range(X_synth.shape[1])])
    df[TARGET_COL] = y_synth
    print("File data.csv tidak ditemukan — memakai dataset sintetis contoh.")
else:
    df = pd.read_csv(PATH_DATA)
    print(f"Data dibaca dari {PATH_DATA}. Shape: {df.shape}")

# -------------------------
# 4) SEDIKIT EDA SINGKAT
# -------------------------
print("\nRingkasan target:")
print(df[TARGET_COL].value_counts())
print("\nTipe kolom:")
print(df.dtypes)

# -------------------------
# 5) SPLIT TRAIN / TEST
# -------------------------
# Gunakan stratified split bila class imbalance agar proporsi class tetap sama.
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)
print(f"\nSplit data: X_train={X_train.shape}, X_test={X_test.shape}")

# -------------------------
# 6) PREPROCESSING
# -------------------------
# Kita tentukan tipe kolom numerik vs kategorikal secara otomatis.
num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

# Transformer untuk numerik: imputasi median + scaling
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Transformer untuk kategorikal: imputasi paling sering + one-hot encode
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])



preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
], remainder="drop")  # drop kolom lain yang tidak disebut

# -------------------------
# 7) PENYEIMBANGAN KELAS (OPTIONAL)
# -------------------------
# Jika dataset imbalance, metode oversampling seperti SMOTE dapat membantu recall/precision.
smote = SMOTE(random_state=42)

# -------------------------
# 8) FITUR SELECTION (OPTIONAL)
# -------------------------
# SelectKBest dengan ANOVA F-test — tunable via hyperparameter k
selector = SelectKBest(score_func=f_classif, k="all")  # default all, akan ditune

# -------------------------
# 9) MODELS & PIPELINE
# -------------------------
# Kita buat beberapa model baseline untuk dibandingkan.
clf_lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
clf_rf = RandomForestClassifier(n_jobs=-1, random_state=42, class_weight="balanced")

# Gunakan imbalanced-learn pipeline agar SMOTE dipanggil hanya pada training folds di CV.
pipeline_lr = ImbPipeline(steps=[
    ("preproc", preprocessor),
    ("select", selector),
    ("smote", smote),
    ("clf", clf_lr)
])

pipeline_rf = ImbPipeline(steps=[
    ("preproc", preprocessor),
    ("select", selector),
    ("smote", smote),
    ("clf", clf_rf)
])

# -------------------------
# 10) EVALUATION FUNCTION (UTILITY)
# -------------------------
def evaluate_model(pipeline, X_test, y_test, label="model"):
    """
    Meng-evaluasi model terlatih pada data test dan mencetak metrik penting.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = None
    try:
        y_proba = pipeline.predict_proba(X_test)[:,1]
    except Exception:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"\n=== Evaluation: {label} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_test, y_proba)
            print(f"ROC AUC: {auc:.4f}")
        except Exception:
            pass
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    return {"accuracy":acc, "precision":prec, "recall":rec, "f1":f1}

# -------------------------
# 11) CROSS-VALIDATED BASELINE
# -------------------------
# Quick CV untuk baseline metrics agar mendapat gambaran.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\nCross-val score baseline (Logistic Regression, f1_macro):")
scores = cross_val_score(pipeline_lr, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
print("F1 scores (folds):", scores)
print("Mean F1:", np.mean(scores))

# -------------------------
# 12) HYPERPARAMETER TUNING (EXAMPLE)
# -------------------------
# Kita tunning beberapa hyperparameter penting dengan RandomizedSearchCV.
param_dist_lr = {
    "select__k": [5, 10, "all"],  # jumlah fitur dipilih
    "clf__C": [0.01, 0.1, 1, 10, 100],  # regulasi LR
    "clf__solver": ["liblinear", "lbfgs"]
}
rand_search_lr = RandomizedSearchCV(pipeline_lr, param_distributions=param_dist_lr,
                                    n_iter=8, scoring="f1", cv=cv, verbose=1, n_jobs=-1,
                                    random_state=42)

print("\nMenjalankan RandomizedSearchCV untuk Logistic Regression...")
rand_search_lr.fit(X_train, y_train)
print("Best params (LR):", rand_search_lr.best_params_)
print("Best CV score (LR):", rand_search_lr.best_score_)

# Tuning RandomForest (contoh Grid)
param_grid_rf = {
    "select__k": [5, 10, "all"],
    "clf__n_estimators": [100, 300],
    "clf__max_depth": [None, 10, 30]
}
grid_search_rf = GridSearchCV(pipeline_rf, param_grid=param_grid_rf,
                              scoring="f1", cv=cv, verbose=1, n_jobs=-1)
print("\nMenjalankan GridSearchCV untuk RandomForest...")
grid_search_rf.fit(X_train, y_train)
print("Best params (RF):", grid_search_rf.best_params_)
print("Best CV score (RF):", grid_search_rf.best_score_)

# -------------------------
# 13) EVALUASI PADA TEST SET
# -------------------------
# Evaluasi model terbaik dari tiap pencarian.
best_lr = rand_search_lr.best_estimator_
best_rf = grid_search_rf.best_estimator_

res_lr = evaluate_model(best_lr, X_test, y_test, label="Best LogisticRegression")
res_rf = evaluate_model(best_rf, X_test, y_test, label="Best RandomForest")

# Pilih model terbaik berdasar metrik yang Anda prioritaskan (misal f1).
best_overall = best_rf if res_rf["f1"] >= res_lr["f1"] else best_lr
print(f"\nModel terbaik berdasarkan F1 di test set: {'RandomForest' if best_overall is best_rf else 'LogisticRegression'}")

# -------------------------
# 14) THRESHOLD TUNING (untuk trade-off precision / recall)
# -------------------------
# Untuk model probabilistik, mengubah threshold bisa meningkatkan precision at cost of recall, atau sebaliknya.
def tune_threshold(pipeline, X, y, thresholds=np.linspace(0.1,0.9,17)):
    proba = pipeline.predict_proba(X)[:,1]
    best = {"threshold": None, "f1": -1, "precision":0, "recall":0}
    for t in thresholds:
        preds = (proba >= t).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        if f1 > best["f1"]:
            best.update({"threshold": t, "f1": f1,
                         "precision": precision_score(y, preds, zero_division=0),
                         "recall": recall_score(y, preds, zero_division=0)})
    return best

if hasattr(best_overall, "predict_proba"):
    thr = tune_threshold(best_overall, X_test, y_test)
    print("\nThreshold tuning result (test set):", thr)

# -------------------------
# 15) CALIBRATION (jika probabilitas tidak kalibrasi)
# -------------------------
# Bila probabilitas yang dihasilkan model tidak akurat, bisa kalibrasi memakai CalibratedClassifierCV.
if not isinstance(best_overall.named_steps["clf"], LogisticRegression):
    try:
        calib = CalibratedClassifierCV(best_overall.named_steps["clf"], cv=3, method="isotonic")
        # Kita butuh pipeline yang hanya menyertakan preproc+select supaya calib fit setelah pipeline transform data
        from sklearn.pipeline import make_pipeline
        simple_pipe = make_pipeline(preprocessor, selector)
        X_train_t = simple_pipe.fit_transform(X_train)
        calib.fit(X_train_t, y_train)
        print("Kalibrasi dilakukan pada classifier.")
    except Exception as e:
        print("Kalibrasi gagal atau tidak perlu:", e)

# -------------------------
# 16) SIMPAN MODEL TERBAIK
# -------------------------
MODEL_OUT = "best_model.joblib"
joblib.dump(best_overall, MODEL_OUT)
print(f"\nModel tersimpan sebagai {MODEL_OUT}")

# -------------------------
# 17) PLOT (CONTOH sederhana)
# -------------------------
try:
    y_pred = best_overall.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha='center', va='center', color='white')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix disimpan sebagai confusion_matrix.png")
except Exception as e:
    print("Gagal membuat plot:", e)

# -------------------------
# 18) CATATAN & TIPS
# -------------------------
# - Jika dataset sangat imbalance: coba gabungkan SMOTE + class_weight, atau coba undersampling pada majority class.
# - Untuk fitur kategorikal high-cardinality: gunakan target encoding (hati-hati leaking) atau embedding (model tree/NN).
# - Untuk model tree (RandomForest / XGBoost): Feature importance bisa dipakai untuk feature selection.
# - Untuk pipeline produksi: simpan preprocessor & selector terpisah sehingga Anda bisa memproses data baru dengan konsisten.
# - Jangan lupa validasi time-based split bila data time-series.
