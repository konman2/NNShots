import pandas as pd
import numpy as np
def normalize(data,norm):
    data[norm]=((data[norm]-data[norm].min())/(data[norm].max()-data[norm].min()))
    return data
df = pd.read_csv("./shot_logs.csv")
all_data = df.to_numpy()
df= df.drop(['LOCATION'],axis=1)
df= df.drop(['PERIOD'],axis=1)
df= df.drop(['GAME_CLOCK'],axis=1)
df =df.fillna(0)
norm = "FINAL_MARGIN"
df = normalize(df,"FINAL_MARGIN")
df = normalize(df,"SHOT_NUMBER")
df= normalize(df,"SHOT_CLOCK")
df = normalize(df,"DRIBBLES")
df = normalize(df,"TOUCH_TIME")
df = normalize(df,"SHOT_DIST")
df = normalize(df,"CLOSE_DEF_DIST")
all_data = df.to_numpy()
# print(df.sample(5))
# print(df.iloc[2])
# print(np.shape(all_data))
np.random.shuffle(all_data)

y = all_data.T[7].T
# print(all_data.T[7])
# print(np.shape(y))
X = all_data.T[0:7].T

start = int(np.shape(X)[0]*0.8)
X_test = X[start:]
X = X[:start]
y_test=y.T[start:].T
y = y.T[:start].T
np.save("./data/X",X)
np.save("./data/X_test",X_test)
np.save("./data/y",y)
np.save("./data/y_test",y_test)


# print(x[2])
# print(y)
