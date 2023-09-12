import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn import svm
import pickle
data = pd.read_csv("creditcard.csv")

x_data = data.drop(columns={"Class", "Time"})
y_data = data["Class"]

print(x_data.head())
print(y_data.head())
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8)

print("Creating kmeans....")
kmeans = KMeans(n_clusters=2, init='k-means++')
kmeans_data = kmeans.fit_transform(X_train)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
pca_kmeans_data = pca.fit_transform(kmeans_data,)

print("Creating svm...")
svmc = svm.SVC(kernel='linear', C=1, cache_size=6000)

svm = GridSearchCV(SVC(probability=True), {'kernel': ['rbf', 'poly'], 'gamma': [1e-3,  1e-4], 'C':[1,10,100,1000]}, cv=5,
                   scoring='%s_macro' % 'precision')
print("fitting...")
svm.fit(kmeans_data, y_train)

means = svm.cv_results_['mean_test_score']
stds = svm.cv_results_['std_test_score']

print("Saving model...")
pickle.dump(svm, open("svm.sav", 'wb'))
# pred = svmc.predict(X_test)
# y_pred_svm = svm.predict(X_test)
# print(svm.best_params_)
