import pandas as pd
import numpy as np
'''NOTE: THIS MATRIX APPEARS TO BE BUGGY'''
#matrix
key = pd.read_csv("random_25_locations.csv")["name"].to_list()
distanceMatrix = pd.read_csv("distance_matrix_with_coordinates.csv")[["Origin Postal Code","Destination Postal Code","Distance (meters)","Duration (seconds)"]]
newDistanceMatrix = np.zeros((len(key),len(key)))
for _, row in distanceMatrix.iterrows():
    newDistanceMatrix[key.index(row["Origin Postal Code"])][key.index(row["Destination Postal Code"])] = row["Distance (meters)"]
pd.DataFrame(newDistanceMatrix).to_csv("newDistanceMatrix.csv")

#location csv
googleCSV = pd.read_csv("random_25_locations.csv")
X=((googleCSV["lat"].values-1.1) *110.574*1000).astype(int)
Y = ((googleCSV["lng"].values-103.6) *111.320*np.cos(np.radians(googleCSV["lat"]))*1000).astype(int)

formatted=np.ones((len(X),5)).astype(int)
formatted[:,0] = np.arange(1,len(X)+1)
formatted[:,2] = X
formatted[:,3] = Y
formatted[:,4] = np.random.rand(1,len(X))*10

formatted = pd.DataFrame(formatted)
formatted.columns = ["NodeNumber","NodeType","X","Y","Demand"]
formatted.to_csv("random_25_locations_FORMATTED.csv")
