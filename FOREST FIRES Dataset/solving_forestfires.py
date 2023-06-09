import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


# Load Forest Fire Dataset
data = pd.read_csv('forestfires.csv')


# Encode the data
data['month'] = data['month']\
    .replace(['jan', 'feb', 'mar', 'apr', 'may', 'jun',
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

data['day'] = data['day']\
    .replace(['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat'],
             [1, 2, 3, 4, 5, 6, 7])


forest = data.values


X = forest[:, 0:12]  # Data
y = data['area']     # Area


# 30% reservado para teste e 70% para treinamento da máquina
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # random_state=3


# The best score
the_best_score = []


# USANDO O NEAR NEIGHBORS REGRESSOR MODEL
scores = []

the_biggest_accuracy = 0
the_biggest_knn = 1

k_range = range(1, 26)

for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)

    knn.fit(X_train, y_train)

    knn_score_in = knn.score(X_test, y_test)

    scores.append(knn_score_in)

    if scores[k-1] > the_biggest_accuracy:
        the_biggest_accuracy = scores[k-1]

        the_biggest_knn = k


knn = KNeighborsRegressor(n_neighbors=the_biggest_knn)

knn.fit(X_train, y_train)

knn_score = round(knn.score(X_test, y_test), 5)

print('Near Neighbors Regressor: ', knn_score)

the_best_score.append(knn_score)


# USANDO O LINEAR REGRESSION MODEL
linreg = LinearRegression()

linreg.fit(X_train, y_train)

linreg_score = round(linreg.score(X_test, y_test), 5)

print('Linear Regression: ', linreg_score)

the_best_score.append(linreg_score)


# USANDO O RANDOM FOREST REGRESSOR MODEL
ranforest = RandomForestRegressor()

ranforest.fit(X_train, y_train)

ranforest_score = round(ranforest.score(X_test, y_test), 5)

print('Random Forest Regressor: ', ranforest_score)

the_best_score.append(ranforest_score)


# USANDO O DECISION TREE REGRESSOR MODEL
dectree = DecisionTreeRegressor()

dectree.fit(X_train, y_train)

dectree_score = round(dectree.score(X_test, y_test), 5)

print('Decision Tree Regressor: ', dectree_score)

the_best_score.append(dectree_score)


# USANDO O SUPPORT VECTOR REGRESSION MODEL -SVR-
svm = SVR()

svm.fit(X_train, y_train)

svm_score = round(svm.score(X_test, y_test), 5)

print('Support Vector Regression: ', svm_score)

the_best_score.append(svm_score)


# Com dados aleatórios
X_new = [[8, 5.6, 3, 1, 99.3, 30.9, 100.87, 3.1, 7.2, 55, 7.7, 0],
         [4, 3.3, 5, 3, 80.76, 25.9, 93.87, 2.1, 4.2, 50, 4.7, 0]]


# Maior score
biggest_score = round(max(float(number) for number in the_best_score), 5)
print('')


# Resolvendo X_new com modelo mais eficiente
if biggest_score == knn_score:
    biggest_score_string = 'Near Neighbors Regressor'
    print('Modelo com maior score: ', biggest_score_string)

    knn_previsao = knn.predict(X_new)

    size_knn = len(knn_previsao)

    for e in range(0, size_knn):
        if knn_previsao[e] < 0:
            knn_previsao[e] = 0

    print('Área prevista com o KNN: ', knn_previsao)


if biggest_score == linreg_score:
    biggest_score_string = 'Linear Regression'
    print('Modelo com maior score: ', biggest_score_string)

    linreg_previsao = linreg.predict(X_new)

    size_liR = len(linreg_previsao)

    for e in range(0, size_liR):
        if linreg_previsao[e] < 0:
            linreg_previsao[e] = 0

    print('Área prevista com o LiR: ', linreg_previsao)


if biggest_score == ranforest_score:
    biggest_score_string = 'Random Forest Regressor'
    print('Modelo com maior score: ', biggest_score_string)

    ranforest_previsao = ranforest.predict(X_new)

    size_RFo = len(ranforest_previsao)

    for e in range(0, size_RFo):
        if ranforest_previsao[e] < 0:
            ranforest_previsao[e] = 0

    print('Área prevista com o RFo: ', ranforest_previsao)


if biggest_score == dectree_score:
    biggest_score_string = 'Decision Tree Regressor'
    print('Modelo com maior score: ', biggest_score_string)

    dectree_previsao = dectree.predict(X_new)

    size_DTr = len(dectree_previsao)

    for e in range(0, size_DTr):
        if dectree_previsao[e] < 0:
            dectree_previsao[e] = 0

    print('Área prevista com o DTr: ', dectree_previsao)


if biggest_score == svm_score:
    biggest_score_string = 'Support Vector Regression'
    print('Modelo com maior score: ', biggest_score_string)

    svm_previsao = svm.predict(X_new)

    size_SVR = len(svm_previsao)

    for e in range(0, size_SVR):
        if svm_previsao[e] < 0:
            svm_previsao[e] = 0

    print('Área prevista com o SVR: ', svm_previsao)
