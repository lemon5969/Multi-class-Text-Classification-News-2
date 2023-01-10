
# Multi class text classification news

I. This project is for categorize article and news to the correct type.

II. The data is from GitHub - https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv

III. This data set 99 duplicate

IV. This project using LSTM, Dense, and Dropout layers.




## Project Output

              precision    recall  f1-score   support

           0       0.84      0.86      0.85       352
           1       0.91      0.81      0.86       263
           2       0.78      0.83      0.81       291
           3       0.95      0.95      0.95       331
           4       0.83      0.84      0.84       251

    accuracy                           0.86      1488
   macro avg       0.86      0.86      0.86      1488
weighted avg       0.86      0.86      0.86      1488

Accuracy score: 0.8622311827956989


## Model Architecture 
![alt text](https://github.com/lemon5969/Multi-class-Text-Classification-News-2/blob/main/Image/model.png?raw=true)

##Training
Epoch 1/100
40/40 [==============================] - 218s 5s/step - loss: 1.5518 - acc: 0.2794 - val_loss: 1.5161 - val_acc: 0.2728

Epoch 2/100
40/40 [==============================] - 188s 5s/step - loss: 1.3096 - acc: 0.4490 - val_loss: 1.0733 - val_acc: 0.5995

Epoch 3/100
40/40 [==============================] - 175s 4s/step - loss: 0.5795 - acc: 0.8038 - val_loss: 0.7748 - val_acc: 0.7030

Epoch 4/100
40/40 [==============================] - 174s 4s/step - loss: 0.2595 - acc: 0.9168 - val_loss: 0.6446 - val_acc: 0.7957

Epoch 5/100
40/40 [==============================] - 189s 5s/step - loss: 0.2257 - acc: 0.9152 - val_loss: 0.6832 - val_acc: 0.7581

Epoch 6/100
40/40 [==============================] - 188s 5s/step - loss: 0.0925 - acc: 0.9733 - val_loss: 0.5249 - val_acc: 0.8555

Epoch 7/100
40/40 [==============================] - 187s 5s/step - loss: 0.0178 - acc: 0.9969 - val_loss: 0.5234 - val_acc: 0.8622

Epoch 8/100
40/40 [==============================] - 173s 4s/step - loss: 0.0024 - acc: 1.0000 - val_loss: 0.5445 - val_acc: 0.8676

Epoch 9/100
40/40 [==============================] - 201s 5s/step - loss: 6.7283e-04 - acc: 1.0000 - val_loss: 0.5610 - val_acc: 0.8669

Epoch 10/100
40/40 [==============================] - 216s 5s/step - loss: 5.3087e-04 - acc: 1.0000 - val_loss: 0.5736 - val_acc: 0.8676

Epoch 11/100
40/40 [==============================] - 176s 4s/step - loss: 4.1208e-04 - acc: 1.0000 - val_loss: 0.5849 - val_acc: 0.8716

Epoch 12/100
40/40 [==============================] - 168s 4s/step - loss: 2.8130e-04 - acc: 1.0000 - val_loss: 0.5938 - val_acc: 0.8690




## Result Graph
![alt text](https://github.com/lemon5969/Multi-class-Text-Classification-News-2/blob/main/Image/tbloggraph.png?raw=true)


## Conclusion
This model can categorize what type of news and articles with accuracy 86%.
## Credits:
This data sets is taken from github @susanli2016 :)


