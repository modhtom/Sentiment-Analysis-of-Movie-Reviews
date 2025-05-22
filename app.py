import nltk
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, confusion_matrix, PrecisionRecallDisplay
import seaborn as sns

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

dataset = load_files("txt_sentoken", categories=["pos", "neg"], encoding="utf-8")
X, y = dataset.data, dataset.target


def preprocess(text):
    text = re.sub(r"[^a-zA-Z']", " ", text).lower()
    tokens = [
        word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)
    ]

    stop_words = set(stopwords.words("english"))
    filtered = [t for t in tokens if t not in stop_words and t.isalpha()]

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos="v") for word in filtered]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vec = TfidfVectorizer(
    ngram_range=(1, 3),
    max_df=0.85,
    min_df=5,
    max_features=7000,
    stop_words=None,
    use_idf=True,
    tokenizer=preprocess,
    sublinear_tf=True,
)
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

models = {
    "logistic_regression": {
        "model": LogisticRegression(max_iter=2000, solver="liblinear"),
        "param_grid": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
        },
    },
    "linear_svc": {
        "model": LinearSVC(dual=True, max_iter=20000),
        "param_grid": {"C": [0.01, 0.1, 1, 10]},
    },
    "random_forest": {
        "model": RandomForestClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 30, 50, None],
            "min_samples_split": [2, 5, 10],
        },
    },
}

for name, model in models.items():
    print(f"\n{name}: ")
    grid = GridSearchCV(
        estimator=model["model"],
        param_grid=model["param_grid"],
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print("Best parameters:", grid.best_params_)
    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds, normalize="true")
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        xticklabels=dataset.target_names,
        yticklabels=dataset.target_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"images/{name}_confusion_matrix.png")
    plt.show()

    if hasattr(best_model, "decision_function"):
        scores = best_model.decision_function(X_test)
    else:
        scores = best_model.predict_proba(X_test)[:, 1]

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, scores)
    plt.title(f"{name} ROC Curve")
    plt.savefig(f"images/{name}_roc_curve.png")
    plt.show()

    # Precision–Recall Curve
    PrecisionRecallDisplay.from_predictions(y_test, scores)
    plt.title(f"{name} Precision-Recall Curve")
    plt.savefig(f"images/{name}_pr_curve.png")
    plt.show()

    # Save Model
    with open(f"model/{name}.pkl", "wb") as f:
        pickle.dump(best_model, f)
