
# coding: utf-8

# In[1]:

import matplotlib
matplotlib.use('Agg')
import json
import pandas as pd
import matplotlib.pyplot as plt


# It seems that 20-60 is the most common length. We can choose to cap at 60 words for our model.

# In[2]:


review=pd.read_csv('review.csv')
review=review.dropna()

# In[14]:


max_len=60


# In[15]:


# split data into train and test
from sklearn.model_selection import train_test_split

X=review['clean_text']
y=review['sentiment']
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42)

X_train=X_train.values
X_test=X_test.values
y_train=y_train.values
y_test=y_test.values


# Words can't be directly fed into the model, instead we need to encode each word to a unique numerical value. Therefore a review is transformed from being an array of words to an array of integer values.

# In[16]:


import numpy as np

# load the glove word vectors

def read_glove_vecs(glove_file):
    with open(glove_file,'r') as f:
        words=set()
        word_to_vec={}
        
        for line in f:
            line=line.strip().split()
            curr_word=line[0]
            words.add(curr_word)
            word_to_vec[curr_word]=np.array(line[1:],dtype=np.float64)
        
        word_to_index={}
        index_to_word={}
        
        i=0
        for w in sorted(words):
            word_to_index[w] = i
            index_to_word[i] = w
            i = i + 1
        
        return word_to_index, index_to_word, word_to_vec    


# In[17]:


word_to_index, index_to_word, word_to_vec= read_glove_vecs('../glove.6B.50d.txt')


# In[18]:


len(word_to_index)


# In[19]:


from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# In[20]:


#convert each sentence to indices:

def sentences_to_indices(X, word_to_index, max_len):
    
    m=X.shape[0]
    
    X_indices=np.zeros((m,max_len))
    
    for i in range(m):
        sentence=X[i].split()
        
        # if a word is not in vocabulary, simply discard it
        j=0
        for w in sentence[0:max_len]:
            if w in word_to_index:
                X_indices[i, j] = word_to_index[w]  
                j=j+1

    return X_indices
            


# In[21]:


sentences_to_indices(X_train[0:2],word_to_index,10)


# In[22]:


# define the embedding layer

def pretrained_embedding_layer(word_to_vec, word_to_index):
    
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    Arguments:
    word_to_vec -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)
    
    """
    vocal_len=len(word_to_index)+1
    vec_len=50  
    
    emb_matrix=np.zeros((vocal_len,vec_len))
    
    for word,index in word_to_index.items():
        emb_matrix[index,:]=word_to_vec[word]
    
    embedding_layer = Embedding(vocal_len,50,trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


# In[23]:


def Sentiment_model(input_shape, word_to_vec, word_to_index):
    
    """
    Function creating the Sentiment Analysis model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (batch_size, max_len,)
    word_to_vec -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    
    sentence_indices = Input(input_shape,dtype='int32')
    embedding_layer= pretrained_embedding_layer(word_to_vec, word_to_index)
    
    embeddings=embedding_layer(sentence_indices)
    
    X = LSTM(6, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(6, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(1)(X)
    X = Activation ('sigmoid') (X)
    
    model = Model(inputs=sentence_indices,outputs=X)
    
    return model


# In[24]:


model = Sentiment_model((max_len,), word_to_vec, word_to_index)
model.summary()


# In[25]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[26]:


X_train_indices = sentences_to_indices(X_train, word_to_index, max_len)
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len)
print(X_train_indices.shape)


# In[27]:


history=model.fit(X_train_indices, y_train, batch_size=50, epochs=20, validation_split=0.3,verbose=2)


# In[28]:


loss, acc = model.evaluate(X_test_indices, y_test)
print()
print("Test accuracy = ", acc)


# In[29]:


print('plot training loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['train_loss','val_loss','train_acc','val_acc'],loc='best')
plt.title('model train vs validation loss and accuracy')
plt.ylabel('loss/acc')
plt.xlabel('epoch')
plt.savefig('../outputs/training_LSTM_2x6.png')

