from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.tokenize import word_tokenize
import nltk

# Descargar recursos necesarios
nltk.download('punkt_tab')
nltk.download('punkt')

class TextClassifier:
    def __init__(self, categories):
        self.categories = categories
        self.newsgroups = None
        self.texts = None
        self.bow_vectorizer = CountVectorizer()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.classifier_bow = LogisticRegression(max_iter=1000)
        self.classifier_tfidf = LogisticRegression(max_iter=1000)

    def load_data(self):
        self.newsgroups = fetch_20newsgroups(subset='train', categories=self.categories, remove=('headers', 'footers', 'quotes'))
        print(f"Número de documentos: {len(self.newsgroups.data)}")
        print(f"Categorías: {self.newsgroups.target_names}")
        print(f"Primer documento:\n{self.newsgroups.data[0]}")

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())  # Minúsculas y tokenización
        tokens = [word for word in tokens if word.isalnum()]  # Eliminar puntuación
        tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]  # Eliminar stopwords
        return ' '.join(tokens)

    def preprocess_data(self):
        self.texts = [self.preprocess_text(doc) for doc in self.newsgroups.data]
        print(self.texts[0])

    def vectorize_data(self):
        bow_features = self.bow_vectorizer.fit_transform(self.texts)
        tfidf_features = self.tfidf_vectorizer.fit_transform(self.texts)
        print(f"Bag-of-Words Shape: {bow_features.shape}")
        print(f"TF-IDF Shape: {tfidf_features.shape}")
        return bow_features, tfidf_features

    def train_and_evaluate(self, features, target, vectorizer_name):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
        if vectorizer_name == 'BoW':
            self.classifier_bow.fit(X_train, y_train)
            y_pred = self.classifier_bow.predict(X_test)
        else:
            self.classifier_tfidf.fit(X_train, y_train)
            y_pred = self.classifier_tfidf.predict(X_test)
        print(f"Evaluación usando {vectorizer_name}:")
        print(f"Precisión: {accuracy_score(y_test, y_pred)}")
        print(classification_report(y_test, y_pred, target_names=self.newsgroups.target_names))

def main():
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    classifier = TextClassifier(categories)
    classifier.load_data()
    classifier.preprocess_data()
    bow_features, tfidf_features = classifier.vectorize_data()
    classifier.train_and_evaluate(bow_features, classifier.newsgroups.target, 'BoW')
    classifier.train_and_evaluate(tfidf_features, classifier.newsgroups.target, 'TF-IDF')

if __name__ == "__main__":
    main()