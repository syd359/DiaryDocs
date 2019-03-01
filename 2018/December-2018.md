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
**Very Clear Explanation**: http://mlwiki.org/index.php/Cumulative_Gain_Chart  
Lift与Cumulative Gain之区别在于除不除当前的sample比例。
```
def lift_chart_plot(yTrue, yPred, ax=None):
    sorted_indices = np.argsort(yPred)[::-1]
    yTrue = yTrue[sorted_indices]
    gains = np.cumsum(yTrue)
    gains = gains / float(np.sum(yTrue)) # 头部（按概率排序）中标/总的真实的True
    
    percentages = np.arange(start=1, stop=len(yTrue) + 1)
    percentages = percentages / float(len(yTrue))

    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])
    percentages = percentages[1:]
    gains2 = gains[1:]

    gains2 = gains2 / percentages  # 计算Lift要除以sample ratio
    
    # 绘图
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
_it is sensitive to differences in both location and shape of the empirical cumulative distribution functions of the two samples._  
> **Empirical distribution function**: This cumulative distribution function is a step function that jumps up by 1/n at each of the n data points. _**cumulative distribution function (CDF)** of a real-valued random variable X, or just distribution function of X, evaluated at x, is the probability that X will take a value less than or equal to x._  

中文的一个解释KS值的：https://www.sohu.com/a/132667664_278472    
KS(Kolmogorov-Smirnov)值越大，  
表示模型能够将正、负客户区分开的程度越大。KS值的取值范围是[0，1]   
通常来讲，KS>0.2即表示模型有较好的预测准确性。  

常用的模型评价还有K-S曲线，它和ROC曲线的画法异曲同工。  
以Logistic模型为例，首先把Logistic模型输出的概率从大到小排序，然后取10%的值（也就是概率值）作为阀值，  
同理把10%*k（k=1,2,3,…,9）处的值作为阀值，  
计算出不同的FPR和TPR值，以10%*k（k=1,2,3,…,9）为横坐标，分别以TPR和FPR的值为纵坐标，就可以画出两个曲线，这就是K-S曲线。  

从K-S曲线就能衍生出KS值，KS=max(TPR-FPR)，即是两条曲线之间的最大间隔距离。  
当(TPR-FPR)最大时，也就是ΔTPR-ΔFPR=0，这和ROC曲线上找最优阀值的条件ΔTPR=ΔFPR是一样的。  

K-S曲线能直观地找出模型中差异最大的一个分段，比如评分模型就比较适合用KS值进行评估；  
但同时，KS值只能反映出哪个分段是区分度最大的，不能反映出所有分段的效果。   

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

## 8. Linux `sed`/`awk`/`perl` replace/convert  
In Python, use `subprocess` or `os` module to use Linux Commend  

1. **`sed`** Linux 原生命令 for string replacement
> `sed -i -e 's/\\N/NaN/g' filename`    

```
import os
os.system(r"sed -i -e 's/\\N/NA/g' test_data.csv")
```
</br>

2. **`awk`** convert string to float  
> `awk '{gsub(/\.?0+$/,"")}1' file`  
</br>

3. **`perl`** for string replacement  
> `perl -i -pe 's/foo/bar/g' ./*`  
`-i` activates in-place editing.
</br>


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
```
%%time
for col in df_a.columns:
    st = time.time()
    pd.to_numeric(df_a[col], errors='ignore').dtypes
    print(col, df_a[col].dtype, time.time() - st)
%%time
for col in df_a.columns:
    st = time.time()
    df_a[col].astype(np.float64, errors='ignore').dtype
    print(col, df_a[col].dtype, time.time()-st)
try:
    import torch
    if torch.cuda.is_available():
        print("Use GPU (Pytorch) for Dtype Convert")
    else:
        print("CUDA is Not Installed Properly")
except ModuleNotFoundError:
    print("Pytorch not Install. Use CPU for Dtype Convert")
```
