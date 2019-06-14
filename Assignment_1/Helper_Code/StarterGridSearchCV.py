#example from: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC(gamma="scale")
#param_grid
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(iris.data, iris.target)
                       
print(clf.cv_results_)