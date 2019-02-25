# 2019-02-14
## 1. PU Learning  
Positive and Unlabeled Learning, also called learning from positive and unlabeled examples.  
> In PU learning, two sets of examples are assumed to be available for training:   
>> The positive set **P** and a mixed set **U**, which is assumed to contain both positive and negative samples, but without these being labeled as such.   
This contrasts with other forms of **semisupervised learning**, where it is assumed that a labeled set containing examples of both classes is available in addition to unlabeled samples. 
A variety of techniques exist to adapt supervised classifiers to the PU learning setting, including variants of the EM algorithm. 

- Useful Doc: https://www.cs.uic.edu/~liub/NSF/PSC-IIS-0307239.html  
- Doc Two: https://roywright.me/2017/11/16/positive-unlabeled-learning/  
  - PU Bagging
  - Two-Step Approaches
  - Positive unlabeled random forest (Not Implemented)
- Github: https://github.com/aldro61/pu-learning (Based On 2008 Paper)
- Github: https://github.com/kiryor/nnPUlearning (Chainer implementation of non-negative PU learning and unbiased PU learning)
  - **Need Pytorch**


## 2. Benefits to use `assert` in Python
>When you are writing code filled with assert statements, you can be more confident that when execution reaches a certain point, certain conditions are guaranteed to be met.

- Link 1: https://www.programiz.com/python-programming/assert-statement  
- Link 2: http://pgbovine.net/programming-with-asserts.htm  

> 
Usage:   
`assert <condition>`  
`assert <condition>, <error message>`  


## 3. PSI, 单变量分析
- 分析feature importance最大的变量，在train，test上的分布（按照bins划分）是否基本一致
```
def cal_bins(classifier, train_data, test_data, num_bins=10, num_top_vars=5):
    
    assert(train_data.shape[0] >= num_bins, "The Given num_bins Larger Than train_data rows")
    assert(len(train_data.columns) >= num_top_vars, "The Given num_top_vars Larger Than train_data columns")
    
    if isinstance(classifier, XGBClassifier):
        top_cols = list(clf.get_booster().get_score(importance_type='gain'))[:num_top_vars]
        
    elif isinstance(classifier, LGBMClassifier):
        top_cols = pd.DataFrame(
            {'columns': lgb_model.booster_.feature_name(), 
              'importance': lgb_model.booster_.feature_importance('gain')
            }).sort_values(by='importance', ascending=False)['columns'][:num_top_vars].tolist()
    else:
        raise ValueError("The Classifier is Not XGBModel or LGBModel")
    
    for col in top_cols:
        print("---- {} ----".format(col))
        # train_data按照num_bins进行均分
        equal_cut = pd.qcut(train_data[col], num_bins)
        # test_data按照train_data的划分，将值划分开
        test_cut = pd.cut(test_data[col], bins=equal_cut.cat.categories)
        print((test_cut.value_counts() / test_data.shape[0] * 100).round(2).astype(str) + '%')
        print('\n\n')
```
