import numpy as np
from PIL import Image

def centreCropResize(imagePath, outPath, imageSize=[256,256]):

    image = Image.open(imagePath).convert('RGB')
    oldImSize = image.size
    minLength = min(oldImSize)
    smallestDim = oldImSize.index(minLength)
    biggestDim = np.setdiff1d([0,1], smallestDim)[0]
    newMaxLength = int((imageSize[0]/oldImSize[smallestDim]) * oldImSize[biggestDim])
    newShape = [0, 0]
    newShape[smallestDim] = imageSize[0]
    newShape[biggestDim] = newMaxLength
    resizedImage = image.resize(newShape)

    left = int((newShape[0] - imageSize[0]) / 2)
    right = newShape[0] - left
    top = int((newShape[1] - imageSize[1]) / 2)
    bottom = newShape[1] - top
    croppedImage = resizedImage.crop((left, top, right, bottom))

    croppedImage.save(outPath)