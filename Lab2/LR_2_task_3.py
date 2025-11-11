import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_iris



iris_dataset = load_iris()

print("Ключі iris_dataset:\n", iris_dataset.keys())
print("\nОпис:\n", iris_dataset['DESCR'])
print("\nНазви відповідей:", iris_dataset['target_names'])
print("\nНазви ознак:\n", iris_dataset['feature_names'])
print("\nТип масиву data:", type(iris_dataset['data']))
print("Форма масиву data:", iris_dataset['data'].shape)
print("\nПерші 5 рядків data:\n", iris_dataset['data'][:5])
print("\nТип масиву target:", type(iris_dataset['target']))
print("Відповіді (мітки):\n", iris_dataset['target'])



url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

print("\nФорма датасету:", dataset.shape)
print("\nПерші 20 рядків:\n", dataset.head(20))
print("\nСтатистичне зведення:\n", dataset.describe())
print("\nКількість прикладів у кожному класі:\n", dataset.groupby('class').size())



dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()



dataset.hist()
pyplot.show()



scatter_matrix(dataset)
pyplot.show()



array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=0.20, random_state=1
)



models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
print("\n=== Оцінювання моделей ===")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))



pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()



model = SVC(gamma='auto')
model.fit(X_train, Y_train)



predictions = model.predict(X_validation)

print("\n=== Оцінка моделі SVM ===")
print("Точність:", accuracy_score(Y_validation, predictions))
print("\nМатриця помилок:\n", confusion_matrix(Y_validation, predictions))
print("\nЗвіт про класифікацію:\n", classification_report(Y_validation, predictions))



X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = model.predict(X_new)
print("Прогноз (мітка):", prediction)
print("Спрогнозований сорт ірису:", prediction[0])
