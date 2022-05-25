import pandas as pd
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

iris_df = pd.read_csv("./iris.csv")
X = iris_df

k = 3


#On prend k centres aléatoirement
centres = X.sample(3).to_numpy()[:,:4]
X = X.to_numpy()

change = True
z = 0
while change and z<1000:
    change = False

    #On initialise 3 partitions
    partitions = []
    for i in range(k):
        partitions.append([])

    #Pour chaque point, on calcule sa distance aux centres
    for data in X:
        distances = []
        for centre in centres:
            distances.append(distance.euclidean(data[:4], centre[:4]))
        partitions[np.argmin(distances)].append(data)

    #On calcule la moyenne pour définir de nouveaux centres
    for i in range(len(partitions)):
        moyenne = np.mean(np.array(partitions[i])[:,:4], axis=0)
        if (moyenne != centre[i]).all():
            change = True
        centres[i] = moyenne 
    z += 1
print(z)

fig = plt.figure()
ax = plt.axes(projection='3d')
for i in range(k):
    p = np.array(partitions[i])
    
    print("Partition {} : \n{} \n\n".format(
        i,
        pd.DataFrame(p)[k+1].value_counts())
    )

    ax.plot3D(
        p[:,0], 
        p[:,1],
        p[:,2],
        c=np.random.rand(3,),
        marker='o',
        linestyle='none'
    )
 
plt.show()
