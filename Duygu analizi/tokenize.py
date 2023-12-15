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
