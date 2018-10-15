# 2018-10-10

### 1. Hive

- Impala 脚本
- HiveQL



### 2. For Personal Fun

- numba: [Compiler that translates a subset of Python and NumPy code into fast machine code](http://numba.pydata.org/)
- Jinja2: [modern and designer-friendly templating language for Python](http://jinja.pocoo.org/docs/2.10/)
- Flask: [microframework for Python based on Werkzeug, Jinja 2 and good intentions](http://flask.pocoo.org/)

- Electron:  [使用 JavaScript, HTML 和 CSS 构建跨平台的桌面应用](https://electronjs.org/)
- PyQt: [Bindings for Qt application framework and runs on all platforms ](https://riverbankcomputing.com/software/pyqt/intro)





# 2018-10-15

### 1. Pandas ---- Flatten JSON fields

- Kaggle: [Quick start: read csv and flatten json fields](https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields/notebook)

**Quick look:**

```python
import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from pandas.io import json

def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

print(os.listdir("../input"))
```

