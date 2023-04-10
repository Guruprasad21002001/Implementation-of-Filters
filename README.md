# Implementation-of-Filters

## Aim:

To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:

Anaconda - Python 3.7 

## Algorithm:

### Step1:

Import cv2, matplotlib.py libraries and read the saved images using cv2.imread().

### Step2:

Convert the saved BGR image to RGB using cvtColor().

### Step3:

By using the following filters for image smoothing:filter2D(src, ddepth, kernel), Box filter,Weighted Average filter,GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]), medianBlur(src, ksize),and for image sharpening:Laplacian Kernel,Laplacian Operator.

### Step4:

Apply the filters using cv2.filter2D() for each respective filters.

### Step5:

Plot the images of the original one and the filtered one using plt.figure() and cv2.imshow().

## Program

```python

### Developed By : Guru Prasad.B
### Register Number : 212221230032

```

### 1. Smoothing Filters:

i) Using Averaging Filter:

```python

import cv2
import numpy as np
import matplotlib.pyplot as plt
image1=cv2.imread('aston.png')
image2=cv2.cvtColor (image1,cv2.COLOR_BGR2RGB) 
kernel = np.ones ((11,11), np.float32)/121
image3=cv2.filter2D(image2,-1, kernel)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1) 
plt.imshow(image2)
plt.title('Orignal') 
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title('Filtered')
plt.axis('off')


```


ii) Using Weighted Averaging Filter:

```python

import cv2
import numpy as np
import matplotlib.pyplot as plt
image1=cv2.imread('aston.png')
image2=cv2.cvtColor (image1,cv2.COLOR_BGR2RGB) 
kernal2 = np.array([[1,2,1],[2,4,2],[1,2,1]])/16 
image3 = cv2.filter2D(image2,-1,kernal2)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1) 
plt.imshow(image2)
plt.title('Orignal') 
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title('Filtered')
plt.axis('off')


```


iii) Using Gaussian Filter:

```python

import cv2
import numpy as np
import matplotlib.pyplot as plt
image1=cv2.imread('aston.png')
image2=cv2.cvtColor (image1,cv2.COLOR_BGR2RGB) 
gaussian_blur=cv2.GaussianBlur(src=image2,ksize=(11,11),sigmaX=0,sigmaY=0)
kernal2 = np.array([[1,2,1],[2,4,2],[1,2,1]])/16 
image3 = cv2.filter2D(image2,-1,kernal2)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1) 
plt.imshow(image2)
plt.title('Orignal') 
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title('Filtered')
plt.axis('off')

```


iv) Using Median Filter:

```python

import cv2
import numpy as np
import matplotlib.pyplot as plt
image1=cv2.imread('aston.png')
image2=cv2.cvtColor (image1,cv2.COLOR_BGR2RGB) 
median=cv2.medianBlur(src=image2, ksize=11)
kernal2 = np.array([[1,2,1],[2,4,2],[1,2,1]])/16 
image3 = cv2.filter2D(image2,-1,kernal2)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1) 
plt.imshow(image2)
plt.title('Orignal') 
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title('Filtered')
plt.axis('off')

```


### 2. Sharpening Filters:

i) Using Laplacian Kernal:

```python

import cv2
import numpy as np
import matplotlib.pyplot as plt
image1=cv2.imread('aston.png')
image2=cv2.cvtColor (image1,cv2.COLOR_BGR2RGB) 
kernel3=np.array([[0,1,0],[1,-4,1],[0,1,0]])
image3=cv2.filter2D(image2,-1, kernel3)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1) 
plt.imshow(image2)
plt.title('Orignal') 
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title('Filtered')
plt.axis('off')

```


ii) Using Laplacian Operator:

```python

import cv2
import numpy as np
import matplotlib.pyplot as plt
image1=cv2.imread('aston.png')
image2=cv2.cvtColor (image1,cv2.COLOR_BGR2RGB) 
new_image = cv2.Laplacian(image2, cv2.CV_64F)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1) 
plt.imshow(image2)
plt.title('Orignal') 
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(new_image)
plt.title('Filtered')
plt.axis('off')

```

## OUTPUT

### 1. Smoothing Filters


i) Using Averaging Filter:

![o1](https://user-images.githubusercontent.com/95342910/230920452-69cdec12-c98f-4e3e-b42a-b3c899fba633.png)


ii) Using Weighted Averaging Filter:


![o2](https://user-images.githubusercontent.com/95342910/230920455-b0a73b14-7636-439d-91e3-57bf6c4095b5.png)

iii) Using Gaussian Filter:

![o3](https://user-images.githubusercontent.com/95342910/230920472-a1dfebb9-01ef-46c7-a20b-178b91970f50.png)


iv) Using Median Filter:

![o4](https://user-images.githubusercontent.com/95342910/230920481-617ebf76-95e9-4922-838e-7b281b4c17d5.png)


### 2. Sharpening Filters:


i) Using Laplacian Kernal:

![o5](https://user-images.githubusercontent.com/95342910/230920485-6cfd772d-0052-4b4c-9b9c-0b0f06833c7a.png)


ii) Using Laplacian Operator:

![o6](https://user-images.githubusercontent.com/95342910/230920490-9c1ac158-a80b-474a-9bd4-cf3d3c45df8b.png)


## Result:

Thus the filters are designed for smoothing and sharpening the images in the spatial domain.


