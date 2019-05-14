# Summary

* [Introduction](README.md)
* [2018-September](2018-september.md)

```
import itertools
import multiprocessing
from multiprocessing import Pool

def cv_test(ps, fs):    
    tmp = "{}, {}".format(ps, fs)
    return tmp
    
a = [str(i) for i in range(10)]
b = list('abscdfghijk')
p = Pool(processes = 4)
start = time.time()
res = p.starmap(cv_test, itertools.product(a,b))
p.close()
p.join()
print('The cross-validation with stratified sampling on 5 cores took time (s): {}'.format(time.time() - start))
print(res)
```
