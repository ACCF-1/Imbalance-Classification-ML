#In[0] setup

import data
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# In[1]

sp_500 = pd.read_csv('D:/Data Science/Time Series Project/saves/raw_data.csv', index_col=0)
sp_500 = sp_500[['S&Ps Common Stock Price Index: Composite','S&Ps Composite Common Stock: Price-Earnings Ratio']]



# In[2]

import seaborn as sns
import matplotlib.pyplot as plt

sp_500['S&Ps Common Stock Price Index: Composite'].plot(figsize=(15,5))

import plotly.express as px
import plotly.graph_objects as go

fig = px.line(sp_500, x=sp_500.index, y="S&Ps Common Stock Price Index: Composite",
              title="Closing Index: Range Slider and Selectors")
fig.update_xaxes(rangeslider_visible=True,rangeselector=dict(
    buttons=list([
        dict(count=1,label="1m",step="month",stepmode="backward"),
        dict(count=6,label="6m",step="month",stepmode="backward"),
        dict(count=1,label="YTD",step="year",stepmode="todate"),
        dict(count=1,label="1y",step="year",stepmode="backward"),
        dict(step="all")])))


fig = go.Figure()
sp_500['Close_M'] = sp_500["S&Ps Common Stock Price Index: Composite"].asfreq('d')
sp_500['Lag_Close_M'] = sp_500['S&Ps Common Stock Price Index: Composite'].asfreq('d').shift(10)
fig.add_trace(go.Scatter(x=sp_500.index, y=sp_500.Close_M, name='Close_M'))
fig.add_trace(go.Scatter(x=sp_500.index, y=sp_500.Lag_Close_M, name='Lag_Close_M'))
fig.show()



# In[000]test

# %%
