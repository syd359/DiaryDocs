# 2019-04-11
## 1. Generalize Regression Pipeline
- [Santander Value Prediction Challenge](https://www.kaggle.com/c/santander-value-prediction-challenge/leaderboard)  
_Predict the value of transactions for potential customers._  

- [Recruit Restaurant Visitor Forecasting](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/overview)  
_Predict how many future visitors a restaurant will receive._  

## 2. KddCup 2019  
- https://www.kdd.org/kdd2019/kdd-cup  
- https://dianshi.baidu.com/competition/29/rule  

## 3. Plotly + Dash VS Bokeh Server
**Plotly + Dash**  
Link: https://plot.ly/d3-js-for-python-and-pandas-charts/  

- **Plotly**
 Github: https://github.com/plotly/plotly.py/  
  _plotly.py is an interactive, open-source, and browser-based graphing library for Python_  
- **Dash** 
 Github: https://github.com/plotly/dash  
 _Analytical Web Apps for Python. No JavaScript Required._  


**Bokeh**  
Link: https://bokeh.pydata.org/en/latest/docs/gallery.html  
Server Tutorial: https://bokeh.pydata.org/en/latest/docs/user_guide/server.html#userguide-server  

## 4. PrettyTable & Textwrap
PrettyTable: https://github.com/jazzband/prettytable  
Textwrap: https://docs.python.org/3/library/textwrap.html  

## 5. Forecasting: Principles and Practice (Time Series Book)
 - Link: https://otexts.com/fpp2/  

  有些Kernel可以借鉴  
 - https://www.kaggle.com/liananapalkova/automated-feature-engineering-for-titanic-dataset  

```
tmp_data = pd.DataFrame({}, columns=pd.date_range(pd.Timestamp(2019,1,1), periods=140))
for item in zip(tmp_data.columns, [1,2,3,4,5,6,7][::-1]*20):
    tmp_data[item[0]] = np.random.uniform(size=num) * item[1]
hist_data = tmp_data.sum(axis=0)
hist_data.head()
tmp_res = pd.Series([np.nan]*hist_data.shape[0], index = hist_data.index)
for i in range(20):
    X = tmp_data.iloc[:, i:i+30]
    print(X.columns[0],'----', X.columns[-1])
    y = tmp_data.iloc[:, i+30]
    print(y.name)
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    X_test = tmp_data.iloc[:, i+1:i+31]
    print(X_test.columns[0],'----', X_test.columns[-1])
    tmp_res.iloc[i+31] = reg.predict(X_test).sum()
    print(tmp_res.index[i+31])
    
    print("\n")

pd.concat([hist_data, tmp_res], axis=1).dropna().plot()
# tmp_res.dropna()
```
