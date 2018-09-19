------

# 2018-09-04

### 1. The Packages to Research:

* **hyperopt**

* **bayes\_opt**

* **TPOT**

* **Auto-sklearn**

Do auto machine learning,**especially auto hyper-parameter tuning**.

Check how the can be used on**lightGBM**. （先看bayes\_opt在lightGBM上的效果）

Benchmark Folder: sx\_zlc



### 2. RNN \(GRU/LSTM\) Model

进一步：

* 0 数据点

* 转换回 实际值 的 误差，以及 比率

* 后三小时的金额 与 前21小时的金额 比值



### 3. GitLab Research ＆Standard

	Already Done !





------

# 2018-09-11

### 1. **数据集 (Kaggle)**

[Home Credit Default Risk (kaggle)](https://www.kaggle.com/c/home-credit-default-risk)

 **第一名**：[https://www.kaggle.com/c/home-credit-default-risk/discussion/64821](https://www.kaggle.com/c/home-credit-default-risk/discussion/64821)  

### 2. AutoML Packages

1. LightGBM官方文档以及测试。 

   官文：[https://lightgbm.readthedocs.io/en/latest/index.html](https://lightgbm.readthedocs.io/en/latest/index.html) 

   Github：[https://github.com/Microsoft/LightGBM](https://github.com/Microsoft/LightGBM)

2. Bayesian Optimization

   - Github: [https://github.com/fmfn/BayesianOptimization](https://github.com/fmfn/BayesianOptimization)
   - 一些关键术语： Gaussian Process -- Upper Confidence Bound (UCB), kappa, maximum of the acquisition function
   - Cubic correlation
   - Proxy Optimization Problem (finding the maximum of the acquisition function)

3. Bayesian Optimization with LGBM

   - 单独使用LGBM：[Good_fun_with_LigthGBM](#)
   - 使用已经训练好的Bayes参数:[LightGBM parameters by Bayesian opt](#)
   - [Simple Bayesian Optimization for LightGBM](#)
   - [[0.797\]LGBM and Bayesian Optimization](#)
   - 对Sklearn里的BayesOpt的一些解释： [https://thuijskens.github.io/2016/12/29/bayesian-optimisation/](https://thuijskens.github.io/2016/12/29/bayesian-optimisation/)

4. Hyperopt (AutoML) 

   官方Docs: [http://hyperopt.github.io/hyperopt/](http://hyperopt.github.io/hyperopt/) 

   Github: [Distributed Asynchronous Hyperparameter Optimization in Python](https://github.com/hyperopt/hyperopt)

   [Automated Model Tuning From Kaggle](#)

   [Home Credit Hyperopt Optimization (Kaggle Kernel)](https://www.kaggle.com/ogrellier/home-credit-hyperopt-optimization/notebook)

5. Spearmint 

   Github: [Spearmint Bayesian optimization codebase](#)

6. MOE 

   Github: [A global, black box optimization engine for real world metric optimization](#)

   官方Docs: [http://yelp.github.io/MOE/](http://yelp.github.io/MOE/)





------

# 2018-09-12

### 1. **Gitlab搭建** 

[通过 docker 搭建自用的 gitlab 服务](https://juejin.im/post/5a4c9ff36fb9a04507700fcc) 

Docker官方：[Docker](https://docs.docker.com/docker-for-windows/install/)

### 2. **Vim使用**



### 3. **"glob" Module**: 

Easy handle folder files

```
for path in glob.glob(r"../data/*.csv", recursive=True):
    logger.info(f"Use Data File ----> {path[8:]}")
```





------

# 2018-09-17

### 1. Hyperopt 自动调参简单教程

 [自动化机器学习超参数调优](https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a)  （含分析）

[Chinese Word Vectors 中文词向量 Github](https://github.com/Embedding/Chinese-Word-Vectors)

> [^]: This project provides 100+ Chinese Word Vectors (embeddings) trained with different **representations** (dense and sparse), **context features** (word, ngram, character, and more), and **corpora**. One can easily obtain pre-trained vectors with different properties and use them for downstream tasks.

### 2. "codecs" Module (Python)

```python
# When you encounter some decode error use pandas like the Following:
"""
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x9c in position 4: invalid start byte
'or' 
UnicodeDecodeError:'utf-8' codec can't decode bytes in position 217813-217814: invalid continuation byte
"""
# You can use codecs modele to Open it and Then put it in a DataFrame

import codecs
import pandas as pd

file_name = r'sample.csv'
fdata = codecs.open(file_name, "r",encoding='utf-8', errors='ignore')
df = pd.read_csv(fdata, header=None, encoding='utf-8', error_bad_lines=False, engine='c')
```

### 3. Get Number of Year/Month/Day from now

```python
# The most Pythonic & easy way from Now
(datetime.now() - df['DateColumn'].astype('timedelta64[M]').astype(np.int)
# change the M to Y, M, D as you want, OR drop np.int to get float years/days/...
# 转化成pd.Timestamp在 year 计算上没问题，但是在months和days上有问题，只能得到日期月份数的差值
```

- **设定Time Zone并返回其月份（1-12）**

`df.dt.tz_localize('Asia/Shanghai').dt.month` 

- **Replace values that have a count smaller than X**

```python
df.loc[df.groupby('A').A.transform('count').lt(2), 'A'] = np.nan  
# if you want larger that 
# just add a ~, such as:
df.loc[~df.groupby('A').A.transform('count').lt(2), 'A'] = np.nan
```



