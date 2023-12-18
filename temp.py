import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("economic_data.csv")
print(data)
print(data.describe())
print(data.head())
print(len(data['Year'].unique())
print(data['Year'])

ls=['afnan','bayan','noor']
df=pd.DataFrame(ls, index=[1,2,3], columns=['sis'])
df.to_csv('mysis.csv')
data=pd.read_csv("mysis.csv")
print(data)

ls=['afnan','bayan','noor']
print(ls)

s=pd.Series(ls)
print(s)

s=pd.Series(ls, index=[1,2,3])
print(s)

print(s.values)
print(s.index)

d=pd.DataFrame(ls)
print(d)

d=pd.DataFrame(ls, index=[1,2,3], columns=["sis"])
print(d)

print(d.values)
print(d.index)

###################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("economic_data.csv")
print(data)

plt.scatter(data["Year"],data["GDP"])
plt.show()

x=data.iloc[:,0]
print(x)
y=data.iloc[:,1]
print(y)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)

plt.scatter(x,y)
plt.plot(x,model.predict(x),'g')

plt.score(x,y)






















