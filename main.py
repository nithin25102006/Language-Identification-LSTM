import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load Dataset
print("\nLoading Dataset...")

df = pd.read_csv("dataset/dataset.csv")

# 2. Handle Missing Values
df = df.dropna(subset=["Text", "language"])

df["Text"] = df["Text"].astype(str).str.strip()
df["language"] = df["language"].astype(str).str.strip()

df = df[
    (df["Text"] != "") &
    (df["language"] != "")
]

df = df.drop_duplicates(subset=["Text", "language"])

print("Dataset Shape:", df.shape)
print("Number of Languages:", df["language"].nunique())

# 3. Text Preprocessing
def preprocess_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text

df["Text"] = df["Text"].apply(preprocess_text)

# 4. Select Manageable Subset
df = (
    df.groupby("language", group_keys=False)
      .sample(n=500, random_state=42)
      .reset_index(drop=True)
)

print("Subset Shape:", df.shape)

# 5. Encode Language Labels
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(df["language"])

num_classes = len(label_encoder.classes_)

print("\nLanguages:", list(label_encoder.classes_))

# 6. Tokenization
MAX_WORDS = 20000
MAX_LEN = 100

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(df["Text"])

sequences = tokenizer.texts_to_sequences(df["Text"])

X = pad_sequences(
    sequences,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

print("Input Shape:", X.shape)

# 7. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# 8. Build LSTM Model
model = Sequential([
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=64,
        mask_zero=True
    ),
    LSTM(32),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nLSTM Model Summary:")
model.summary()

# 9. Train Model
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

print("\nTraining LSTM Model...")

history = model.fit(
    X_train,
    y_train,
    epochs=8,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# 10. Evaluate Model
print("\nEvaluating Model...")

probabilities = model.predict(X_test, verbose=0)

y_pred = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(num_classes),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

# 11. Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(num_classes)
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

plt.figure(figsize=(14, 10))

sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title("Language Identification - Confusion Matrix")
plt.xlabel("Predicted Language")
plt.ylabel("Actual Language")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.show()

# 12. Training Accuracy Graph
plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("LSTM Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/accuracy.png", dpi=150)
plt.show()

# 13. Training Loss Graph
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("LSTM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/loss.png", dpi=150)
plt.show()

# 14. Save Model and Tokenizer
model.save("models/language_model.keras")

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nModel and tokenizer saved successfully!")

# 15. Predict Five Unseen Text Samples
unseen_texts = [
    "Hello, how are you today?",
    "Bonjour, comment allez-vous aujourd'hui?",
    "Hola, como estas hoy?",
    "Guten Morgen, wie geht es dir?",
    "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"
]

unseen_sequences = tokenizer.texts_to_sequences(unseen_texts)

unseen_padded = pad_sequences(
    unseen_sequences,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

predictions = model.predict(unseen_padded, verbose=0)

print("\n========== UNSEEN TEXT PREDICTIONS ==========")

for text, prediction in zip(unseen_texts, predictions):

    predicted_index = np.argmax(prediction)

    language = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = prediction[predicted_index] * 100

    print("\nText:", text)
    print("Predicted Language:", language)
    print(f"Confidence: {confidence:.2f}%")

print("\nPROJECT COMPLETED SUCCESSFULLY!")