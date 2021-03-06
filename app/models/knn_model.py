import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from app.models.dataset.data_preprocessing import preproc_mnist

x_train, y_train, x_test, y_test = preproc_mnist()
x_train, x_test = x_train.reshape((60000, 28*28)), x_test.reshape((10000, 28*28))

pca = PCA(n_components=0.95)
knn_clf = KNeighborsClassifier(n_neighbors=5)
model = Pipeline([('pca', pca), ('knn_clf', knn_clf)])
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Accuracy: ", accuracy_score(y_pred, y_test))

joblib.dump(model, "trained/knn_model.joblib", compress=3)