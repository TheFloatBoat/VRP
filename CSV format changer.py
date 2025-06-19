import pandas as pd
import numpy as np
key = pd.read_csv("random_25_locations.csv")["name"].to_list()
distanceMatrix = pd.read_csv("distance_matrix_with_coordinates.csv")[["Origin Postal Code","Destination Postal Code","Distance (meters)","Duration (seconds)"]]
newDistanceMatrix = np.zeros((len(key),len(key)))
for _, row in distanceMatrix.iterrows():
    newDistanceMatrix[key.index(row["Origin Postal Code"])][key.index(row["Destination Postal Code"])] = row["Distance (meters)"]
pd.DataFrame(newDistanceMatrix).to_csv("newDistanceMatrix.csv")


#todo:automate this
googleCSV = pd.read_csv("random_25_locations.csv")
X=googleCSV["lat"].values *110.574
Y = googleCSV["lng"].values *111.320*np.cos(np.radians(googleCSV["lat"]))

print(X)
print(Y)
