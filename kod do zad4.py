import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# W ramki wpisuję dane z tabeli testowej kolumna po kolumnie
test = {
    'Sepal_length': [6.3, 5.4, 6.0, 5.0, 6.0, 4.4, 6.8, 5.4, 6.7, 5.2],
    'sepal_width': [3.3, 3.7, 2.2, 2.3, 2.2, 2.9, 2.8, 3.0, 3.1, 4.1],
    'petal_length': [6.0, 1.5, 4.0, 3.3, 5.0, 1.4, 4.8, 4.5, 5.6, 1.5],
    'petal_width': [2.5, 0.2, 1.0, 1.0, 1.5, 0.2, 1.4, 1.5, 2.4, 0.1],
    'decyzja': ['Iris-virginica', 'Iris-setosa', 'Iris-virginica', 'Iris-setosa',
                 'Iris-virginica', 'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor',
                 'Iris-setosa', 'Iris-virginica']
}

test1 = pd.DataFrame(test)

# Rozdzielam baze na argumenty i decyzje
X = test1[['Sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = test1['decyzja']

sasiedzi = [1, 3, 5]
odleglosc = ['cityblock', 'euclidean', 'minkowski']
metoda_glosowania = ['uniform', 'distance']

# Iteracja po wszystkich kombinacjach parametrów
srednia_dokladnosc = 0
a = {}
b = {}
c = {}
for n in sasiedzi:
    for d in odleglosc:
        for w in metoda_glosowania:
            # Tworzymy klasyfikator kNN z pomocą zaimportowanego KNeighborsClassifier z aktualnymi parametrami
            knn = KNeighborsClassifier(n_neighbors=n, metric=d, weights=w)

            # Ocena klasyfikatora za pomocą walidacji krzyżowej (3-krotna walidacja)
            wynik = cross_val_score(knn, X, y, cv=3)
            sredni_w = wynik.mean()

            # Jeśli uzyskano lepszy wynik, zaktualizuj najlepsze parametry
            if sredni_w > srednia_dokladnosc:
                srednia_dokladnosc = sredni_w
                a = {'sasiedzi': n}
                b = {'odleglosc': d}
                c = {'metoda_glosowania': w}

print(a)
print(b)
print(c)
print(srednia_dokladnosc)