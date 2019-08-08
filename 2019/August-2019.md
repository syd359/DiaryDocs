# 2019-08-08
1. JupyterLab 一些简单的命令
```
%%bash
echo scikit-learn > requirements.txt

# Install a pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install -r requirements.txt
```
