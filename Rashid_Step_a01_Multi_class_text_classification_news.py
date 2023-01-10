#%%
#Library
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,confusion_matrix
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import pickle
import json
import os
from modules import text_cleaning, lstm_model_creation
#%%
#Data Path
TB_LOGS_PATH = os.path.join(os.getcwd(),'TB_TextClassificationProject',datetime.datetime.now().strftime('%Y%m%d-%H%M%S')) #TensorBoard Log Path
URL = "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv" #Data Frame URL PATH
#%%
#Step 1) Data Loading

df = pd.read_csv(URL)
#%%
#Step 2) Data Inspection
display (df.describe())
display (df.info())
display (df.head(10))

#%%
#TO check duplicated
display (df['text'].duplicated().sum())# found 99 duplicated

#To check NaNs
display (df.isna().sum())
#%%
print(df['text'][1])

#%%
# Step 3) Data Cleaning
# Thing to be removed
for index, temp in enumerate(df['text']):
    df['text'][index] = text_cleaning(temp)
#%%
df = df.drop_duplicates() # remove duplicate 99 duplicates data
#%%
#Step 4) Features Selection
X = df['text']
y = df['category']

#%%
#Step 5) Data Preprocessing
#Tokenizer
# Need to identify via checking the unique words in the sentence
num_words = 6000
tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV') #OOV stand for out of vocabulary
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

# Padding
X = pad_sequences(X, maxlen=350, padding='post', truncating='post')

#%%
# OHE 
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y[::,None])

#%%
#Train Test Split
X_train, X_test, y_train,y_test = train_test_split(X, y, train_size=0.3,shuffle=True,random_state=123)

#%%
#Model Development
model = lstm_model_creation(num_words, y.shape[1])
display (plot_model(model,show_layer_names=(True),show_shapes=(True)))
#%%
#TensorBoard
tb_callback = TensorBoard(log_dir=TB_LOGS_PATH)
es_callback = EarlyStopping(monitor='val_loss',patience=5,verbose=0,restore_best_weights=True)

#%%

hist = model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=16,epochs=100, callbacks=[tb_callback,es_callback])

#%%
#Model Analysis

plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['training','validation'])
plt.show()

#%%

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['training','validation'])
plt.show()
#%%
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)

#%%
print (classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
acc_score = accuracy_score(y_test, y_pred)
disp.plot()
print("Accuracy score: " + str(acc_score))
#%%

#%%
#Model Saving
# to save trained model
model.save("model.h5")

# to save one hot encoder model

with open ("ohe.pkl",'wb') as f:
  pickle.dump(ohe,f)

# tokenizer
token_json = tokenizer.to_json()
with open ("tokenizer.json",'w') as f:
  json.dump(token_json,f)
# %%


#%%