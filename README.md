## Trend Line Drawer!
an specific Trend Line Draw for financial Markets in python
## Installation
Simple! Clone or download it. install requ and pass it a dataframe like example in main file
```python
pip install plotly
pip install numpy
pip install pandas
```
## How its Work?
it has a lot options like degree limit on sides. check validation of trend define by you, and you can see al detailes on tags
```python
df = pd.read_csv("xauusd.csv", header=0,
                 usecols=["close", 'high', 'low', 'time', 'open', 'time'])
df = df[-1000:].reset_index()
model = {
    "pivot_connect1": 2,
    "counter_limit1": 100,
    "degree_limit1": 1,
    "distance_limit1": 10,
    "countinues_limit1": 3,
    "swing_percentage1": 0.03,
    "degree_side_limit1": "up",
    "band_perc1": 0.007,
    "band_increase_step1": 0.000025,
}
_,_,_,_=draw_trendline(model,df['close'],df['high'],df['low'],df['open'],df.index,True)
```
## Overview
<p align="center">
    <a href="_" target="_blank">
    <img src="https://github.com/safaritrader/Trend-Line/blob/main/Screenshot%202023-11-10%20195423.jpg">
</a></p>
<p align="center">
    <a href="_" target="_blank">
    <img src="https://github.com/safaritrader/Trend-Line/blob/main/Screenshot%202023-11-10%20195443.jpg">
</a></p>

---

## Contact
you can email me or text me :
email : info@global-fxs.com
telegram : +989137070309
