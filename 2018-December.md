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
```

## 7. KS Value
**Kolmogorov-Smirnov** statistic on 2 samples.  
_This is a two-sided test for the **null hypothesis:** 2 independent samples are drawn from the same continuous distribution_
```
def ks_value(data1, data2):
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / (1.0*n1)
    cdf2 = np.searchsorted(data2, data_all, side='right') / (1.0*n2)
    d = np.max(np.absolute(cdf1 - cdf2))
    return d
```
```
dump(svm_clf, 'svm_clf')
dump(lgb_clf, 'lgb_clf')
dump(xgb_clf, 'xgb_clf')
dump(rf_clf, 'rf_clf')

if os.path.exists('svm_clf'):
    clf1 = load('svm_clf')
    print(str(clf1.__class__).split('.')[-1].split('\'')[0])
    print(clf1.predict(X_test)[0])
if os.path.exists('lgb_clf'):
    clf1 = load('lgb_clf')
    print(str(clf1.__class__).split('.')[-1].split('\'')[0])
    print(clf1.predict(X_test)[0])
if os.path.exists('xgb_clf'):
    clf1 = load('xgb_clf')
    print(str(clf1.__class__).split('.')[-1].split('\'')[0])
    print(clf1.predict(X_test)[0])
        
```
