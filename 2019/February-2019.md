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
