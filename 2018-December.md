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
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
# import scikitplot as skplt
import matplotlib.pyplot as plt


def lift_chart_plot(yTrue, yPred, ax=None):
    
    sorted_indices = np.argsort(yPred)[::-1]
    yTrue = yTrue[sorted_indices]
    gains = np.cumsum(yTrue)

    percentages = np.arange(start=1, stop=len(yTrue) + 1)

    gains = gains / float(np.sum(yTrue))
    percentages = percentages / float(len(yTrue))

    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])

    percentages = percentages[1:]
    gains2 = gains[1:]

    gains2 = gains2 / percentages

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.set_title('Lift Chart', fontsize=15)

    ax.plot(percentages, gains2, lw=3, label='Class 1')

    ax.plot([0, 1], [1, 1], 'k--', lw=2, label='Baseline')

    ax.set_xlabel('Percentage of sample', fontsize=10)
    ax.set_ylabel('Lift', fontsize=10)
    ax.tick_params(labelsize=10)
    ax.grid('on')
    ax.legend(loc='lower right', fontsize=10)

    return ax


if __name__ == "__main__":
    X, y = make_classification(n_samples=10000, n_features=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    svm_clf = SVC(probability=True)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    y_pred_proba = svm_clf.predict_proba(X_test)

    # skplt.metrics.plot_lift_curve(y_test, y_pred_proba)
    # plt.show()
    lift_chart_plot(y_test, y_pred_proba[:, 1])
    plt.show()
```
