# model.py
import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model():
    categories = [
        'rec.sport.baseball',
        'sci.space',
        'talk.politics.mideast',
        'comp.sys.mac.hardware'
    ]

    data = fetch_20newsgroups(subset="all", categories=categories, remove=("headers", "footers", "quotes"))

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_df=0.9, min_df=5, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    # Save model + labels
    joblib.dump(model, "news_model.pkl")
    joblib.dump(data.target_names, "labels.pkl")

    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()
