# -*- coding: utf-8 -*-
"""LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CcOWavED5cnRXP1pvdpdTaVoUrCLuRAE
"""

import pandas as pd
import nltk
import tensorflow as tf
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow import sparse
from tensorflow.sparse import reorder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from keras.utils import to_categorical

# Load the 'interpress_news_category_tr_lite' dataset
import numpy as np
dataset = pd.read_csv("/content/veriseti.csv")
random_rows = np.random.choice(dataset.index, size=len(dataset)//6, replace=False)

# Seçilen satırları veri çerçevesinden sil
df = dataset.drop(random_rows)

# Veri kümesini karıştırın (shuffle)
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Veri setinin yüzde 80'ini eğitim, yüzde 20'sini test olarak kullanın
train_size = int(0.8 * len(dataset))
train_data, test_data = dataset[:train_size], dataset[train_size:]

import tensorflow as tf
from tensorflow.sparse import SparseTensor

# Pre-Processing and Bag of Word Vectorization using Count Vectorizer
nltk.download('punkt')
nltk.download('stopwords')

# Turkish language model
nltk.download('stopwords')
nltk.download('words')

stop_words = set(stopwords.words('turkish'))

token = RegexpTokenizer(r'\w+')

stop_words_list = list(stop_words)
cv = CountVectorizer(stop_words=stop_words_list, ngram_range=(1, 1), tokenizer=token.tokenize, token_pattern=None)

text_counts = cv.fit_transform(dataset['Icerik'])

# Splitting the data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(text_counts, dataset['Kategori'], test_size=0.25, random_state=5)
X_train_coo = X_train.tocoo()

X_train_sparse = tf.sparse.SparseTensor(
    indices=tf.cast(tf.stack([X_train_coo.row, X_train_coo.col], axis=1), tf.int64),
    values=tf.constant(X_train_coo.data, dtype=tf.float32),
    dense_shape=tf.constant(X_train_coo.shape, dtype=tf.int64)
)
# Convert X_train to a SparseTensor and reorder
X_train_reordered = tf.sparse.reorder(X_train_sparse)
X_train_dense = tf.sparse.to_dense(X_train_reordered)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
# Model Building
model = Sequential()
model.add(Embedding(100, 20))
model.add(SpatialDropout1D(0.3))
model.add(SimpleRNN(64, dropout=0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

from sklearn.preprocessing import LabelEncoder

# LabelEncoder'ı tanımla
label_encoder = LabelEncoder()

# Eğitim ve test etiketlerini sayısallaştır
Y_train_encoded = label_encoder.fit_transform(Y_train)
Y_test_encoded = label_encoder.transform(Y_test)

# One-hot encode Y_train ve Y_test
Y_train_one_hot = to_categorical(Y_train_encoded, num_classes=6)
Y_test_one_hot = to_categorical(Y_test_encoded, num_classes=6)

# Model Training
# Model Training
model.fit(X_train_reordered, Y_train_one_hot, epochs=10, batch_size=16, verbose=1)

# Model Testing
model.evaluate(X_test, Y_test)

"""Yeni model"""

import pandas as pd
import numpy as np
import nltk
import tensorflow as tf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow import sparse
from tensorflow.sparse import reorder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from keras.utils import to_categorical

# Load the 'interpress_news_category_tr_lite' dataset
dataset = pd.read_csv("/content/veriseti.csv")
dataset.head(10)
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

dataset['Icerik'] = dataset['Icerik'].replace(np.nan,'',regex=True)

sentiment_scores = []
for text in dataset['Icerik']:
        scores = sid.polarity_scores(text)
        sentiment_scores.append(scores['compound'])

dataset['Sentiment Score'] = sentiment_scores

def get_sentiment_label(score):
    if score >= 0.05:
      return 'Pozitif'
    elif score <= -0.05:
      return 'Negatif'
    else:
      return 'Nötr'

sentiment_labels = dataset['Sentiment Score'].apply(get_sentiment_label)

dataset['Sentiment Label'] = sentiment_labels

import matplotlib.pyplot as plt

sentiment_counts = dataset['Sentiment Label'].value_counts()

plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
plt.title('Duygu Dağılımı')
plt.show()

print(dataset['Kategori'].unique())

import matplotlib.pyplot as plt

categories = ['dunya', 'ekonomi', 'genel', 'guncel', 'kultur-sanat', 'magazin']
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

for i, category in enumerate(categories):
    data = grouped[grouped['Kategori'] == category]

    # Check if data is empty before plotting
    if not data.empty:
        ax = axes[i // 3, i % 3]
        ax.pie(data['Adet'], labels=data['Sentiment Label'], autopct='%1.1f%%')
        ax.set_title(f'{category} Kategorisi için Duygu Sütunu ')
    else:
        print(f'{category} Kategorisi için Veri Yok ')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for enhanced plotting capabilities

# Assuming dataset is a DataFrame containing the necessary columns

fig = plt.figure(figsize=(8, 6))

# Use seaborn for better visualization options
sns.barplot(data=grouped, x='Kategori', y='Adet', hue='Sentiment Label')
plt.title('Duygu Sütunu vs Kategori')
plt.xlabel('Kategori')
plt.ylabel('Adet')
plt.show()