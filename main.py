
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

# Cargar dataset
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))

# Explorar datos
print(f"Número de documentos: {len(newsgroups.data)}")
print(f"Categorías: {newsgroups.target_names}")
print(f"Primer documento:\n{newsgroups.data[0]}")

# Preprocesamiento de texto
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Minúsculas y tokenización
    tokens = [word for word in tokens if word.isalnum()]  # Eliminar puntuación
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]  # Eliminar stopwords
    return ' '.join(tokens)

# Aplicar preprocesamiento
texts = [preprocess_text(doc) for doc in newsgroups.data]

print(texts[0])


# Bag-of-Words
bow_vectorizer = CountVectorizer()
bow_features = bow_vectorizer.fit_transform(texts)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_features = tfidf_vectorizer.fit_transform(texts)

print(f"Bag-of-Words Shape: {bow_features.shape}")
print(f"TF-IDF Shape: {tfidf_features.shape}")

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    bow_features, newsgroups.target, test_size=0.25, random_state=42
)

# Entrenar un clasificador con el vectorizador BoW
classifier_bow = LogisticRegression(max_iter=1000)
classifier_bow.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_bow = classifier_bow.predict(X_test)

# Evaluar el modelo
print("Evaluación usando BoW:")
print(f"Precisión: {accuracy_score(y_test, y_pred_bow)}")
print(classification_report(y_test, y_pred_bow, target_names=newsgroups.target_names))

# Repetir el proceso con TF-IDF
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(
    tfidf_features, newsgroups.target, test_size=0.25, random_state=42
)

# Entrenar el clasificador con el vectorizador TF-IDF
classifier_tfidf = LogisticRegression(max_iter=1000)
classifier_tfidf.fit(X_train_tfidf, y_train_tfidf)

# Predecir en el conjunto de prueba
y_pred_tfidf = classifier_tfidf.predict(X_test_tfidf)

# Evaluar el modelo
print("Evaluación usando TF-IDF:")
print(f"Precisión: {accuracy_score(y_test_tfidf, y_pred_tfidf)}")
print(classification_report(y_test_tfidf, y_pred_tfidf, target_names=newsgroups.target_names))