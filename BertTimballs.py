import pandas as pd
import torch
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # links
    text = re.sub(r"@\w+|#", '', text)  # menções e hashtags
    text = re.sub(r"[^\w\s]", '', text)  # pontuação
    text = re.sub(r"\d+", '', text)  # números
    return text.strip()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_excel("base_Categorizada.xlsx")

# Pré-processamento
df["Text"] = df["Text"].astype(str).apply(preprocess)

# Codificação dos rótulos
le = LabelEncoder()
df["Sentimento_Label"] = le.fit_transform(df["Sentimento_Label"])

# Divisão treino/teste
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Text"].tolist(), df["Sentimento_Label"].tolist(), test_size=0.2, random_state=1
)

# Tokenizador BERTimbau
tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

def tokenize_function(texts):
    return tokenizer(texts["Text"], padding="max_length", truncation=True)

train_dataset = Dataset.from_dict({"Text": train_texts, "labels": train_labels})
val_dataset = Dataset.from_dict({"Text": val_texts, "labels": val_labels})

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Carregamento do modelo
model = BertForSequenceClassification.from_pretrained(
    "neuralmind/bert-base-portuguese-cased", 
    num_labels=len(le.classes_)
).to(device)

# Argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Treinamento
trainer.train()

predictions = trainer.predict(val_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_test = predictions.label_ids

# Exibir relatório de classificação
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Acurácia:", accuracy_score(y_test, y_pred))
