#Loading necessary modules/packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Load dataframe and feature vectors already created
df = pd.read_csv('data\interim\processed_text.csv', index_col=0)
X = pd.read_csv('data\interim\\feature_vecs.csv')

#Scaling features vectors
scaler_x = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)

#Scaling target vector
scaler_y = MinMaxScaler()
y_scaled = []

for set_id in range(1,9):
    essay_scores = df.loc[df['essay_set']==set_id,'domain1_score'].values.reshape(-1,1)
    scaled_scores = scaler_y.fit_transform(essay_scores)
    y_scaled.append(scaled_scores)

y_scaled = np.concatenate(y_scaled).flatten()

#Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)
