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


# 2019-02-27
## 1. Population Stability Index (PSI)
- **Population Stability Index**  
> The population stability index simply indicates changes in the population of loan applicants. However, this may or may not result in deterioration in performance of the scorecard to predict risk. Nevertheless, the PSI indicates changes in the environment which need to be further investigated  

  Explanation：http://ucanalytics.com/blogs/population-stability-index-psi-banking-case-study/
  
  Rules：  
  Less than 0.1	 ---- Insignificant change ----	  No action required  
  0.1 – 0.25	   ---- Some minor change	   ----   Check other scorecard monitoring metrics  
  Greater than 0.25	---- Major shift in population ---- Need to delve deeper  

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

## 2. Weight of Evidence & Information Value  
**Clear Explaination**: https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html  
**Kinda More Details**: https://multithreaded.stitchfix.com/blog/2015/08/13/weight-of-evidence/  
- **WoE**   
  _The weight of evidence tells the predictive power of an independent variable in relation to the dependent variable._
  > The value of WoE will be 0 if the odds of Relative Frequency of Goods / Relative Frequency Bads is equal to 1.   
If the Relative Frequency of Bads in a group is greater than the Relative Frequency of Goods, the odds ratio will be less than 1 and the WoE will be a negative number;   
If the Relative Frequency of Goods is greater than the Relative Frequency of Bads in a group, the WoE value will be a positive number.     

- **IV**  
_Information value is one of the most useful technique to select important variables in a predictive model.  
It helps to rank variables on the basis of their importance._  

  Less than 0.02 ----	Not useful for prediction  
  0.02 to 0.1 ----	Weak predictive Power  
  0.1 to 0.3 ----	Medium predictive Power  
  0.3 to 0.5 ----	Strong predictive Power  
  \>0.5	 ---- Suspicious Predictive Power  
```
def weight_of_evidence(X, y, var, num_bins=10):
    import warnings
    from sklearn.utils.multiclass import type_of_target
    
    assert(type_of_target(y) == 'binary', "The target is Not Binary") 
    ix0 = pd.Series(y).value_counts().index[0]
    ix1 = pd.Series(y).value_counts().index[1]
    
    count0 = (y.values == ix0).sum()
    count1 = (y.values == ix1).sum()
    
    assert(var in X.columns, "The variable(var) not in X columns")
    if type_of_target(X[var]) == 'continuous':
        equal_cut = pd.qcut(X[var], q=num_bins)
        ix0_cut = pd.cut(X[var][y.values == ix0], 
                         bins=equal_cut.cat.categories).value_counts()
        ix1_cut = pd.cut(X[var][y.values == ix1], 
                         bins=equal_cut.cat.categories).value_counts()
        
    elif type_of_target(X[var]) == 'multiclass':
        warnings.warnings('\nVariable is Categorical Dtype, This implement of WoE may not correct')
        ix0_cut = X[var][y.values == ix0].value_counts()
        ix1_cut = X[var][y.values == ix1].value_counts()
        
    else:
        raise ValueError("The Variable Dtype is {}, which is confusing".format(type_of_target(X[var])))
    
    tmp_df = pd.concat([ix0_cut, ix1_cut], axis=1)
    tmp_df.columns = ['y=={}'.format(ix0), 'y=={}'.format(ix1)]
    tmp_df['y=={} percent'.format(ix0)] = tmp_df['y=={}'.format(ix0)] / count0
    tmp_df['y=={} percent'.format(ix1)] = tmp_df['y=={}'.format(ix1)] / count1

    tmp_df['WOE'] = np.log(tmp_df['y=={} percent'.format(ix0)] / tmp_df['y=={} percent'.format(ix1)])
    
    tmp_df['IV_row'] = ((tmp_df['y=={} percent'.format(ix0)] - tmp_df['y=={} percent'.format(ix1)]) * tmp_df['WOE'])
    tmp_df['IV'] = tmp_df['IV_row'].sum()

    print(tmp_df)
  ```

## 3. Time Decorator for time measurement   
```
import time
from functools import wraps
from time import time
import inspect


def _format_time(timespan, precision=3):
    """Formats the timespan in a human readable form"""

    if timespan >= 60.0:
        # we have more than a minute, format that in a human readable form
        # Idea from http://snipplr.com/view/5713/
        parts = [("d", 60*60*24), ("h", 60*60), ("min", 60), ("s", 1)]
        time_lst = []
        leftover = timespan
        for suffix, length in parts:
            value = int(leftover / length)
            if value > 0:
                leftover = leftover % length
                time_lst.append(u'%s%s' % (str(value), suffix))
            if leftover < 1:
                break
        return " ".join(time_lst)
    else:
        return "{0:.{1}f}".format(timespan, precision)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # print('[%r] args:[%r, %r] took: %s sec' % (f.__name__, args, kw, _format_time(te-ts)))
        print('[%r] args: [%s] took: %s sec' % (f.__name__, str(inspect.signature(f)), _format_time(te - ts)))
        # str(inspect.signature(sssa)
        return result
    return wrap


# An example
@timing
def f(a, key=None):
    for _ in range(a):
        i = 0
    return -1


if __name__ == "__main__":
    # st = time.clock()
    # time.sleep(5)
    # end = time.clock()
    #
    # cpu_user = end - st
    # print("CPU times: user %s" % (_format_time(cpu_user)))
    for i in range(50):
        if i < 10 or i > 40:
            print(i)
        elif i == 10:
            print('...')
```


