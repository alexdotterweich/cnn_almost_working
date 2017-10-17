import re
import os
from math import sqrt
import numpy as np
from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2

FULL_PATH= ""
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
l2_lambda = 0.0001


def remove_special_chars(word):
    return re.sub('[^A-Za-z0-9]+', ' ', word)

# Building word embeddings from the pretrained GloVe model
embeddings_index = {}
f = open(os.path.join(FULL_PATH, "glove.42B.300d.txt"))
for line in f:
    values = line.split() 
    word = values[0]
    coefs = np.asarray(values[1:], dtype="float32")
    embeddings_index[word] = coefs
f.close()

print("Found %s word vectors in glove" % len(embeddings_index))

# Preparing our own dataset 
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
buisness_categories = []
label_mapping = {}
f = open(os.path.join(FULL_PATH, "Workbook2.csv"))
first_line = True
for row in f:
    if not first_line:
        cells = row.split(",")
        if cells[1] != "":
            if cells[1] in label_mapping:
                label_mapping[cells[1]] += 1
            else:
                label_mapping[cells[1]] = 1
            texts.append(remove_special_chars(cells[0]))
    first_line = False

# Replace the words in label dict with nums 0-num labels corresponding to elements in list
# Positive there is a better way to do this - will refactor once it all works
i = 0
for label in label_mapping: 
    labels_index[label] = i
    for num_key in range(label_mapping.get(label)):
        labels.append(i)
    i+=1

print("Found %s words in our csv file" % len(texts))

# Vectorize the words
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print("Found %s unique words" % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Create one-hot vectors for each of the labels
labels = to_categorical(np.asarray(labels))

print("Shape of data: ", data.shape)
print("Shape of label: ", labels.shape)

# Split the data into a training set and a validation set
# Will later perform 10 fold crossvalidation to find best split

# Shufling the len(data) items of data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

# Data (x) and labels (y) are actually split here
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]


print("Preparing our embedding matrix")
# word_not_found keeps track of words that aren't found in the 
# GloVe word embeddings. A lot of these words can be parsed 
# to find a meaninful word still - this is another possible refactoring
# Ex. "EDISONPARKFAST" isn't found in GloVe, but "Edison", "Park", and "Fast" might be

word_not_found = 0
num_words = min(MAX_NB_WORDS, len(texts))

# For some reason, initializing with zeros works better than initializing
# with any of the other 2 variations of initialization with small random nums
# Might just be confirmation bias though 
embedding_matrix = np.random.randn(num_words, EMBEDDING_DIM)/sqrt(num_words)
# embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
# embedding_matrix = np.random.randn(num_words, EMBEDDING_DIM)/sqrt(2/num_words)

for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is None:
        word_not_found += 1
    if embedding_vector is not None:
        # Words not found in embedding index will == None
        embedding_matrix[i] = embedding_vector
print(embedding_matrix.shape)

# Finally, load pre-trained word embeddings into an Embedding layer
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


print("number of words not found: ", word_not_found)
print("Finally training model")

# Shape of input (X, 1000)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
embedded_sequences = embedding_layer(sequence_input)

# val_acc is still low, so it means it's not super great at generalizing new things
# looping k=4 times through changing validation data to subset each time


seed = 7
np.random.seed(seed)
kfold = KFold(n_splits=4, random_state=seed)

# Have to actually update history for the kfold crossvalidation to work, 
# but for right now, I only look at one iteration in the interest of time
history = []
for train_indices, val_indices in kfold.split(data):
    print(num_validation_samples, " many num_validation_samples")
    convs = []
    # Yoon Kim paper - multiple filter sizes applied yields higher accuracy
    # These filter sizes were chosen to mimic another paper that expanded on 
    # how to implement the Yoon Kim one
    filters = [3,4,5]
    for f in filters:
        x = Conv1D(nb_filter=128,filter_length=f,activation='elu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        convs.append(x)

    x_train, x_val = data[train_indices], data[val_indices]
    y_train, y_val = labels[train_indices], labels[val_indices]

    x = Conv1D(128, 5, activation='elu', kernel_regularizer=l2(l2_lambda))(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='elu',kernel_regularizer=l2(l2_lambda))(x)
    # I've been experimenting with different levels of dropout to optimize the val_acc fluctuations
    # x = Dropout(0.2)(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='elu',kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)
    x = Dense(128, activation='elu',kernel_regularizer=l2(l2_lambda))(x)
    preds = Dense(len(labels_index), activation='softmax')(x)
    model = Model(sequence_input, preds)

    # Lower momentum slows the oscillation of the val_acc in the more closely related label csv
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.05, nesterov=True)

    print("Training new iteration on " + str(x_train.shape[0]) + " training samples, " 
            + str(x_val.shape[0]) + " validation samples")
    
    # print(model.summary())
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['acc'])

    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=22, batch_size=128, shuffle=True, validation_split=VALIDATION_SPLIT)