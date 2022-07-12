
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

def logEcc(nPix):

    X = nPix
    base = 10
    newCoords = np.empty(X, dtype=int)
    eccs = np.linspace(-1 + 1 / X, 1 - 1 / X, X)
    for x in range(X):
        e = eccs[x]
        if e == 0:
            newCoords[x] = x
        else:
            newCoords[x] = X / 2 - math.log(base, np.abs(e)) * e / np.abs(e)
    newCoords = np.array(newCoords / np.max(newCoords) * X / 2 + X / 2, dtype=int)-1
    return(newCoords)

inImage = np.array(Image.open('/home/dave/Datasets/sample.jpg'))
X,Y,Z = inImage.shape
newX = logEcc(X)
newY = logEcc(Y)

outData = np.empty(shape=(X,Y,Z))
for x in range(X):
    for y in range(Y):
        outData[x,y,:] = inImage[newX[x],newY[y],:]

outImage = Image.fromarray(outData.astype('uint8'), 'RGB')
outImage.save('/home/dave/Datasets/sample2.jpg')

plt.plot(list(range(X)), newX)
plt.xlabel='eccentricity'
plt.ylabel='log eccentricity'
plt.savefig('/home/dave/Datasets/mapping.jpg')