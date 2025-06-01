import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Pré-processamento
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # links
    text = re.sub(r"@\w+|#", '', text)  # menções e hashtags
    text = re.sub(r"[^\w\s]", '', text)  # pontuação
    text = re.sub(r"\d+", '', text)  # números
    return text.strip()

# Carregar os dados
df = pd.read_excel("base_Categorizada.xlsx")

# Aplicar o pré-processamento
df['Text'] = df['Text'].astype(str)

# Processar rótulos multirrótulo
df['Sentimento_Label'] = df['Sentimento_Label'].apply(lambda x: x.split(';'))
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['Sentimento_Label'])

# Vetorização dos textos
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(df['Text'])

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=1)

# Treinar modelo Naive Bayes com estratégia One-vs-Rest
model = OneVsRestClassifier(MultinomialNB(alpha=1.0))
model.fit(X_train, y_train)

# Previsões
y_pred = model.predict(X_test)

# Relatório de classificação
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
print("Acurácia:", accuracy_score(y_test, y_pred))
