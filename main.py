from keras.datasets import imdb
from keras import layers
from keras import models
from keras import metrics
from keras import losses
import numpy as np

# -----------------------------Load IMDB----------------------------------
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000
)
# -------------------------------Model------------------------------------
model = models.Sequential()
model.add(layers.Dense(16,
                       activation='relu',
                       input_shape=(10000,)))
model.add(layers.Dense(16,
                       activation='relu'))
model.add(layers.Dense(1,
                       activation='sigmoid'))
# ----------------------------Compilation----------------------------------
model.compile(optimizer='rmsprop',
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
# -----------------------------Decode review-------------------------------
"""
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]
)
decoded_review = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]]
)
print(decoded_review)
"""
# ------------------------binary matrix encoding-------------------------


def vectorize_sequences(sequences, dimention=10000):
    results = np.zeros((len(sequences), dimention))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# ----------------------------vectorize labels---------------------------
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
# ----------------------------Decode reviews------------------------------
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]
# ------------------------------Train model-------------------------------
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
model.fit(x_train,
          y_train,
          epochs=4,
          batch_size=512)
results = model.evaluate(x_test, y_test)
