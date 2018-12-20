# 2018-12-4  
## 1. Three Modules to Rock  
 - argparse: [Parser for command-line options, arguments and sub-commands](https://docs.python.org/3/library/argparse.html)  
 - Dask: [Dask is a flexible library for parallel computing in Python](https://docs.dask.org/en/latest/)  
 - Numba: [Numba - JIT compiler that translates a subset of Python and NumPy code into fast machine code](http://numba.pydata.org/)  

## 2. Useful Python Modules by Example  
 **THE PYTHON 3 STANDARD LIBRARY BY EXAMPLE**  
 - Link: https://pymotw.com/3/  
 
## 3. Stacking  
 - A kaggler notebook: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard  
 - A Python Module: http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/  
   Github: https://github.com/rasbt/mlxtend

## 4. Famous AutoML Frameworks
 - H2O: https://github.com/h2oai/h2o-3  
 - auto_ml: [Automated machine learning for analytics & production](https://github.com/ClimbsRocks/auto_ml)
 - tpot: [A Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming](https://github.com/EpistasisLab/tpot)  
 - auto_sklearn: [Automated Machine Learning with scikit-learn](https://github.com/automl/auto-sklearn)  
 - **Auto Feature Engineering**: [An open source python framework for automated feature engineering](https://github.com/Featuretools/featuretools)  

## 5. Cython Learning
 - Github: https://github.com/cython/cython  

## 6. Lift Chart Plot
```
def lift_chart_plot(yTrue, yPred):
    """
    Plot lift chart:
    X-->False Positive Rate
    y-->LIFT Score

    :param yTrue:
    :param yPred:
    :return:
    """
    fpr, _, thresholds = roc_curve(yTrue, yPred)
    print(thresholds.shape)
    lift_score = np.array([(precision_score(yTrue,
                                            np.where(yPred < threshold, 0, 1)) * yTrue.shape[0] / yTrue.sum())
                           for threshold in thresholds])
    fig = plt.figure(figsize=(10, 10))
    plt.plot(fpr[5::10], lift_score[5::10])
    plt.xlabel("False Positive Rate")
    plt.ylabel("LIFT")
    plt.title("Lift Chart")
    plt.show()


if __name__ == "__main__":
    X, y = make_classification(n_samples=10000, n_features=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    svm_clf = SVC(probability=True)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    y_pred_proba = svm_clf.predict_proba(X_test)[:, 1]

    lift_chart_plot(y_test, y_pred_proba)
```
