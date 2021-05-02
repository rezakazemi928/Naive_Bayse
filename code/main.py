import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from NaiveBayse import GussianNB

features, labels = load_wine(return_X_y = True)
print('Dataset has been loaded')

model_selection = KFold(n_splits = 3, random_state = 45, shuffle = True)

for train_indx, test_indx in model_selection.split(features):
    x_train, x_test = features[train_indx], features[test_indx]
    y_train, y_test = labels[train_indx], labels[test_indx]
print('Dataset has been splited with KFold method')

model = GussianNB(x_train, y_train, x_test)
mean, var = model.fit()
predictions, probs_list = model.predict(mean, var)
print('Prediction stage has been completed.')

predictions_array = np.array(predictions)
probs_array = np.array(probs_list)

print(f'Acuracy: {accuracy_score(y_true = y_test, y_pred = predictions_array)}')
print(confusion_matrix(y_test, predictions_array))

# ROC visualization
fpr, tpr, thresh = roc_curve(y_test, predictions_array, pos_label = 2)
gmeans = np.sqrt(tpr * (1-fpr))
ix = np.argmax(gmeans)
plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='Naive_bayse')
plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()