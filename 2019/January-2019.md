# 2019-01-08  
## 1. Python Package Structure  
- The Hitchhiker’s Guide to Python: https://docs.python-guide.org/  
- A tour on Python Packaging: https://manikos.github.io/a-tour-on-python-packaging  
- Sample repository structure:      https://github.com/kennethreitz/samplemod  
- License Type: https://choosealicense.com/  
- A Human's Ultimate Guide to `setup.py`:  https://github.com/kennethreitz/setup.py  

## 2. Coding Style  
- PEP 8 — the Style Guide for Python Code:  https://pep8.org/  

## 3. A Sample For Fully use __func__ method example  
**Vector Class** 
> Detail tests and explains are in **Fluent Python** p261  
> Full implementation in p289  
```
from array import array
import math

class Vector2d:
    typecode = 'd'

    def __init__(self, x, y):
        self.__x = float(x)
        self.__y = float(y)
    
    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    def __iter__(self):
        return (i for i in (self.x, self.y))
    
    def __repr__(self):
        class_name = type(self).__name__
        return '{}({!r}, {!r})'.format(class_name, *self) # {'!r'} just for str representation
    
    def __str__(self):
        return str(tuple(self))
    
    def __bytes__(self):
        return (bytes([ord(self.typecode)]) + 
                bytes(array(self.typecode, self)))
    
    def __eq__(self, other):
        return tuple(self) == tuple(other)
    
    def __hash__(self):
        return hash(self.x) ^ hash(self.y)

    def __abs__(self):
        return math.hypot(self.x, self.y)
    
    def __bool__(self):
        return bool(abs(self))
    
    def angle(self):
        return math.atan2(self.y, self.x)
    
    def __format__(self, fmt_spec=''):
        if fmt_spec.endswith('p'):
            fmt_spec = fmt_spec[:-1]
            coords = (abs(self), self.angle())
            outer_fmt = '<{}, {}>'
        else:
            coords = self
            outer_fmt = '({}, {})'
        components = (format(c, fmt_spec) for c in coords)
        return outer_fmt.format(*components)
    
    @classmethod
    def frombytes(cls, octets):
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        return cls(*memv)
```

## 4. How to get all table definitions in a database in Hive?
You can do this by writing a simple bash script and some bash commands.  

First, write all table names in a database to a text file using:  
`$hive -e 'show tables in <dbname>' | tee tables.txt`  
Then create a bash script (describe_tables.sh) to loop over each table in this list:  
```
while read line
do
 echo "$line"
 eval "hive -e 'describe <dbname>.$line'"
done
```
Then execute the script:  
`$chmod +x describe_tables.sh`  
`$./describe_tables.sh < tables.txt > definitions.txt`  
The definitions.txt file will contain all the table definitions.  

# 2019-01-16  
## 1. Lime  
> Github: https://github.com/marcotcr/lime  

_Lime support H2o, so here FINALLY I only use H2o framework as my referance._  
**2. H2o**
- Github: [H2o Python Verion](https://github.com/h2oai/h2o-3/tree/master/h2o-py)  
- Docs: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html  
- Tutorial: https://github.com/h2oai/h2o-tutorials  

## 2. Print text with color  
Answers:  
http://ozzmaker.com/add-colour-to-text-in-python/  
https://stackoverflow.com/questions/16816013/is-it-possible-to-print-using-different-color-in-ipythons-notebook  

On the default Python prompt:
```
>>> print("\x1b[31m\"red\"\x1b[0m")
"red"
```
In the notebook:
```
In [28]: print("\x1b[31m\"red\"\x1b[0m")
         "red"
```

## 3. Optuna
**A hyperparameter optimization framework, particularly designed for machine learning.**  
- Github: https://github.com/pfnet/optuna  
- Docs: https://optuna.org  

**Key Features**
- Parallel distributed optimization
- Pruning of unpromising trials
- Web dashboard

https://stackoverflow.com/questions/50558849/python-how-to-trace-function-execution-order-in-large-project

# 2019-01-26
## 1. Some useful Pandas Groupby usage
`df.groupby(['col_name'])['other_col'].transform('mean')`  // transform other_col values to mean group by colname  
`df.groupby(['col_name']).agg({'purchase_amount': ['sum', 'mean', 'max', 'min', 'std']})` // agg col groupy by colname  
`df.groupby(['col_name']).agg({'col_name': ['size']})`  
**`df.groupby(['col_name'])['other_col'].value_counts().unstack()`**  

## 2. Factorization Machines (FM)  
**field-aware factorization machines (FFM)**
Article: https://www.analyticsvidhya.com/blog/2018/01/factorization-machines/  
中文的介绍: https://blog.csdn.net/john_xyz/article/details/78933253  

## 3. Xlearn
> **Implement FFM & FM**  
_xLearn is a high performance, easy-to-use, and scalable machine learning package, including linear model (LR), factorization machines (FM), and field-aware factorization machines (FFM), which can be used to solve large-scale machine learning problems. xLearn is especially useful for solving machine learning problems on large-scale sparse data._

**Github:** https://github.com/aksnzhy/xlearn  
**Docs:** https://xlearn-doc.readthedocs.io/en/latest/index.html  


