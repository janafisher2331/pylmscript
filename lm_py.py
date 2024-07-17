#!/usr/bin/env python
# coding: utf-8

# # Linear Model in Python

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# # Load Dataset

# In[ ]:

df = pd.read_csv("regrex1.csv")

# In[ ]:

df.head()

# # Create Scatter Plot

# In[ ]:

plt.scatter(df['x'],df['y'])
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.savefig("pylm_scatter.png", format="png")

# # Create Linear Model of Data

# In[ ]:

model=LinearRegression()

# In[ ]:

x = np.array(df['x']).reshape(-1,1)
y = np.array(df['y'])

# In[ ]:

model.fit(x,y)

# In[ ]:

r_sq = model.score(x,y)
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")

# In[ ]:

y_pred = model.predict(x)

# # Plot Linear Model

# In[ ]:

plt.scatter(x,y)
plt.plot(x,y_pred)
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.savefig("pylm_fit.png", format="png")
