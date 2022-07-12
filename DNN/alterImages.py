import numpy as np
import glob
from PIL import Image
import random
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import kornia
import math

def occludeImages(images=None, method=None, coverage=.5, colour=(0,0,0), invert=False, propOccluded=1.0):

    r"""Adds occluders to image.

        Arguments:
            image (tensor):
            method (string): type of occlusion to apply. Options include:
                             barHorz (horizontal bars), polkadot, orientedNoise
            coverage (float, range 0:1): proportion of image to be occluded
            colour (list of tuples each of length 3, range 1:255): RGB colour for occluded pixels
            invert (bool): invert the occlusion pattern or not.

        Returns:
            occluded image (tensor)"""

    occludedImages = torch.zeros_like(images)
    H,W = images.shape[2:4]

    for i in range(images.shape[0]):

        # get coverage
        if coverage is None:
            thisCoverage = -1
            coveragePath = ''
        else:
            if type(coverage) == list:
                thisCoverage = random.choice(coverage)
            else:
                thisCoverage = coverage
            coveragePath = f'{int(thisCoverage * 100)}/'


        # if no occlusion desired
        if thisCoverage == 0 or np.random.uniform() > propOccluded:
            occludedImages[i, :, :, :] = images[i, :, :, :]

        # if occlusion desired
        else:

            image = images[i, :, :, :].permute(1,2,0)
            imagePIL = Image.fromarray(np.array(image*255, dtype=np.uint8)).convert('RGBA')

            # select occlusion image
            if type(method) == list:
                thisOccType = random.choice(method)
            else:
                thisOccType = method
            if thisOccType in ['naturalTextured', 'naturalTextured2']:
                occluderPaths = glob.glob(f'DNN/images/occluders/{thisOccType}/*.png')
            else:
                occluderPaths = glob.glob(f'DNN/images/occluders/{thisOccType}/{coveragePath}*.png')
            occluderPath = random.choice(occluderPaths)

            # get occ image as tensor
            occluderPIL = Image.open(occluderPath).convert('RGBA')
            occluderPIL = occluderPIL.rotate(np.random.randint(21)-10) # randomly rotate between +- 10 degrees

            # ensure image and occluder are same size
            if imagePIL.size != occluderPIL.size:
                occluderPIL = occluderPIL.resize(imagePIL.size)

            # if occluder is textured, paste occluder over image
            if thisOccType in ['naturalTextured', 'naturalTexturedCropped', 'naturalTextured2', 'naturalTexturedCropped2']:
                imagePIL.paste(occluderPIL, (0,0), occluderPIL)
                occludedImage = torch.tensor(np.array(imagePIL.convert('RGB')))
                occludedImages[i, :, :, :] = occludedImage.permute(2,0,1)/255

            # if occluder is untextured, set colour and paste over image
            else:
                # empty final occluder image
                occluderColoured = np.zeros((H, W, 4), dtype=np.uint8)

                # fill first 3 channels with fill colour
                if type(colour) == list:
                    fillCol = torch.tensor(random.choice(colour))
                elif type(colour) == tuple:
                    fillCol = torch.tensor(colour)
                occColourPIL = Image.new(color=tuple(fillCol),mode='RGB',size=(H,W))
                occluderColoured[:, :, :3] = np.array(occColourPIL)

                # fill last channel with alpha layer of occluder
                occluderAlpha = torch.tensor(np.array(occluderPIL))[:, :, 3]  # load image, put in tensor
                occluderAlphaInv = 1 - occluderAlpha  # get inverse of image
                if invert:
                    occluderColoured[:,:,3] = occluderAlphaInv
                else:
                    occluderColoured[:,:,3] = occluderAlpha

                occluderColouredPIL = Image.fromarray(occluderColoured, mode = 'RGBA')

                # paste occluder over image
                occludedImagePIL = imagePIL.copy()
                occludedImagePIL.paste(occluderColouredPIL, (0,0), occluderColouredPIL)
                occludedImage = torch.tensor(np.array(occludedImagePIL.convert('RGB')))
                occludedImages[i, :, :, :] = occludedImage.permute(2,0,1)/255 # rescale to range(0,1)

    return(occludedImages)

'''
dispImage = images[0,:,:,:]
#dispImage = occImage
dispImage = image.permute(1, 2, 0)
dispImage = np.array(dispImage).astype(np.uint8)
plt.imshow(dispImage)
plt.show()
'''

def addNoise(images, ssnr):
    noised_images = torch.zeros_like(images)
    normalize = transforms.Normalize(mean=[0.449], std=[0.226])
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for i in range(images.size(0)): # Batch size
        image = images[i, :, :, :]

        #### Gaussian noise
        sigma = (1 - ssnr) / 2 / 3
        signal = (image-0.5) * ssnr + 0.5
        noise = np.tile(np.random.normal(0, sigma, (1, images.size(2), images.size(3))), (images.size(1), 1, 1))
        noise = torch.from_numpy(noise).float().to(images.device)
        noised_image = signal + noise
        noised_image[noised_image > 1] = 1
        noised_image[noised_image < 0] = 0
        noised_image = normalize(noised_image)
        noised_images[i] = noised_image

        #### Fourier noise
        # image_fft = np.fft.fft2(np.mean(image.cpu().detach().np(), axis=0))
        # image_fft_phase = np.angle(image_fft)
        # # np.random.shuffle(image_fft_phase) # wrong! np.random.shuffle only works with the first axis of array
        # np.random.shuffle(image_fft_phase.flat) # correct way to do it
        # image_fft_shuffled = np.multiply(image_fft_avg_mag, np.exp(1j * image_fft_phase))
        # image_recon = abs(np.fft.ifft2(image_fft_shuffled))
        # image_recon = (image_recon - np.min(image_recon)) / (np.max(image_recon) - np.min(image_recon))
        #
        # signal = (image - 0.5) * ssnr + 0.5
        # noise = np.tile((image_recon - 0.5) * (1 - ssnr), (images.size(1), 1, 1))
        # noise = torch.from_np(noise).float().to(images.device)
        # noised_image = signal + noise
        # noised_image[noised_image > 1] = 1
        # noised_image[noised_image < 0] = 0
        # noised_image = normalize(noised_image)
        # noised_images[i] = noised_image

    noised_images = noised_images.repeat(1, 3, 1, 1) # RGB
    return noised_images


def blurImages(images, sigmas, weights):

    blurred_images = torch.zeros_like(images)
    for i in range(images.size(0)): # Batch size
        image = images[i, :, :, :]

        weights = np.asarray(weights).astype('float64')
        weights = weights / np.sum(weights)
        sigma = np.random.choice(sigmas, 1, p=weights)[0]
        kernel_size = 2 * math.ceil(2.0 * sigma) + 1

        if sigma == 0:
            blurred_image = image
        else:
            blurred_image = kornia.filters.gaussian_blur2d(torch.unsqueeze(image, dim=0), kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
        blurred_images[i] = blurred_image

    return blurred_images
