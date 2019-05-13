# Summary

* [Introduction](README.md)
* [2018-September](2018-september.md)

```
import itertools
import multiprocessing
from multiprocessing import Pool
cv_index = [(i, j) for i, j in sss.split(X, y)]
params = list(itertools.product(cv_index, classifiers))
def cv_test(params):
    global X
    global y
    train_index = params[0][0]
    test_index = params[0][1]
    clf = params[1]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    name = clf.__class__.__name__
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_probas = clf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probas[:,1])
    auc_score = auc(fpr, tpr)
    return [name, acc, loss, auc_score]
p = Pool(processes = 4)
start = time.time()
res = p.map(cv_test, params)
p.close()
p.join()
print('The cross-validation with stratified sampling on 5 cores took time (s): {}'.format(time.time() - start))
```
