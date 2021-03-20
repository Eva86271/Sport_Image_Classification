```python
import numpy as np
import pandas as pd
import os
import shutil
import glob
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns          
import matplotlib.pyplot as plt             
import cv2
from PIL import Image 
import tensorflow as tf
import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
from tensorflow.keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Dense,Dropout,Input,BatchNormalization,Flatten
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

```


```python
print(tf.__version__)
```

    2.4.0
    


```python
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pathlib import Path
trainPath=r'C:\Users\subha\Downloads\Sport\train'
validatePath=r'C:\Users\subha\Downloads\Sport\validate'
testPath=r'C:\Users\subha\Downloads\Sport\test'
trainPathmode=r'C:\Users\subha\Downloads\Game\train'
validatePathmode=r'C:\Users\subha\Downloads\Game\validate'
testPathmode=r'C:\Users\subha\Downloads\Game\test'

```


```python
list_of_class=[]
for ent in os.scandir(r'C:\Users\subha\Downloads\Sport\train'):
    list_of_class.append(ent.name)
```


```python
extension_list=[]
ext_dict={}
for path in [trainPath,validatePath,testPath]:
    for ent in os.scandir(path):
       for file in os.listdir(path+"\\"+ent.name):
        namefile=file.split('.')
        if(namefile[1] in ext_dict.keys()):
            ext_dict[namefile[1]]=ext_dict[namefile[1]]+1
        else:
            ext_dict[namefile[1]]=0
         
        
```


```python
print("Filename",ext_dict.keys())
print("Values",ext_dict.values())
```

    Filename dict_keys(['jpg', 'png', 'JPG', 'jpeg', 'aspx', 'ashx', 'PNG', 'cms', 'gif', 'webp', 'Jpeg'])
    Values dict_values([2007, 84, 16, 84, 1, 3, 4, 0, 2, 2, 0])
    


```python

if(os.path.isdir(r'C:\Users\subha\Downloads\Game\train\\'+list_of_class[0])) is False:
    for i in range(len(list_of_class)):
     os.mkdir(r'C:\Users\subha\Downloads\Game\train'+"\\"+list_of_class[i])
     os.mkdir(r'C:\Users\subha\Downloads\Game\test'+"\\"+list_of_class[i])
     os.mkdir(r'C:\Users\subha\Downloads\Game\validate'+"\\"+list_of_class[i])
else:
   print("The respective directories",list_of_class,"does exists")
```

    The respective directories ['badminton', 'boxing', 'chess'] does exists
    


```python
#saving converted files in directories

def cleanfiles(segment,background):
    if(segment=="train"):
        background.save(trainPathmode+"\\"+ent.name+"\\"+namefile[0]+".jpg")
    elif(segment=="test"):
        background.save(testPathmode+"\\"+ent.name+"\\"+namefile[0]+".jpg")
    else:
        background.save(validatePathmode+"\\"+ent.name+"\\"+namefile[0]+".jpg")
    
```


```python
count=0
file_list=[]
for path in [trainPath,validatePath,testPath]:
    for ent in os.scandir(path):
        ext=path.split("\\")
        track=ext[len(ext)-1]
        for file in os.listdir(path+"\\"+ent.name):  
         size_file=os.stat(path+"\\"+ent.name+"\\"+file)
         file_list.append(path+"\\"+ent.name+"\\"+file)
         if (size_file.st_size > 0):
           namefile=file.split('.')
           rgba_image = Image.open(path+"\\"+ent.name+"\\"+file)
           rgba_image.load()
           if(file.endswith("jpg")): 
              im_resized = rgba_image.resize((224, 224))  
              cleanfiles(track,im_resized)
           else:
              background = rgba_image.convert("RGB")
              im_resized = background.resize((224, 224))
              cleanfiles(track,im_resized)
         else:
           os.remove(path+"\\"+ent.name+"\\"+file)
           count=count+1
    print("File removed:",count)     
```

    File removed: 0
    File removed: 0
    File removed: 0
    


```python
select_file=file_list[0]
for i in [100,150,200,224,256]:
    rgba_image = Image.open(select_file)
    resized=rgba_image.resize((i,i))
    print("Image with size (",i,"X",i,")")
    plt.imshow(resized)
    plt.axis("off")
    plt.show()
   
           
```

    Image with size ( 100 X 100 )
    


![png](output_9_1.png)


    Image with size ( 150 X 150 )
    


![png](output_9_3.png)


    Image with size ( 200 X 200 )
    


![png](output_9_5.png)


    Image with size ( 224 X 224 )
    


![png](output_9_7.png)


    Image with size ( 256 X 256 )
    


![png](output_9_9.png)



```python

trainAugment =ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True)
validateAugment=ImageDataGenerator(rescale=1./255)
```


```python
train_generator=trainAugment.flow_from_directory(
    trainPathmode,
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=True
    )
```

    Found 1475 images belonging to 3 classes.
    


```python
valid_generator=validateAugment.flow_from_directory(
    validatePathmode,
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    )
```

    Found 326 images belonging to 3 classes.
    


```python
imgs,labels=next(train_generator)
```


```python
def plotImages(img_arr):
    fig,axes=plt.subplots(1,10,figsize=(20,20))
    axes=axes.flatten()
    for im,ax in zip(img_arr,axes):
      ax.imshow(im)
      ax.axis("off")
    plt.tight_layout()
    plt.show()
       
```


```python
plotImages(imgs)
print(labels)
```


![png](output_15_0.png)


    [[1. 0. 0.]
     [1. 0. 0.]
     [1. 0. 0.]
     [0. 0. 1.]
     [0. 1. 0.]
     [1. 0. 0.]
     [1. 0. 0.]
     [0. 0. 1.]
     [0. 1. 0.]
     [0. 1. 0.]
     [1. 0. 0.]
     [1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]
     [1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]
     [1. 0. 0.]
     [0. 1. 0.]
     [0. 1. 0.]
     [0. 0. 1.]
     [0. 1. 0.]
     [1. 0. 0.]
     [1. 0. 0.]
     [1. 0. 0.]
     [0. 1. 0.]
     [1. 0. 0.]
     [1. 0. 0.]
     [1. 0. 0.]
     [1. 0. 0.]
     [0. 1. 0.]
     [0. 1. 0.]]
    


```python
baseModel=tf.keras.applications.ResNet50(
    include_top=False,
     weights="imagenet",
    input_tensor=Input(shape=(224,224,3))
)
```


```python
for bl in baseModel.layers:
    bl.trainable=False
```


```python
headModel=baseModel.output
headModel=MaxPooling2D(pool_size=(3,3))(headModel)
headModel=Dense(512,activation="relu")(headModel)
headModel=MaxPooling2D(pool_size=(2,2))(headModel)
headModel=Dense(200,activation="relu")(headModel)
headModel=Dense(100,activation="relu")(headModel)
headModel=Flatten(name="flatten")(headModel)
headModel=Dropout(rate=0.6)(headModel)
headModel=Dense(3,activation="softmax")(headModel)
```


```python
model=Models.Model(inputs=baseModel.input,outputs=headModel)
```


```python

model.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_3 (InputLayer)            [(None, 224, 224, 3) 0                                            
    __________________________________________________________________________________________________
    conv1_pad (ZeroPadding2D)       (None, 230, 230, 3)  0           input_3[0][0]                    
    __________________________________________________________________________________________________
    conv1_conv (Conv2D)             (None, 112, 112, 64) 9472        conv1_pad[0][0]                  
    __________________________________________________________________________________________________
    conv1_bn (BatchNormalization)   (None, 112, 112, 64) 256         conv1_conv[0][0]                 
    __________________________________________________________________________________________________
    conv1_relu (Activation)         (None, 112, 112, 64) 0           conv1_bn[0][0]                   
    __________________________________________________________________________________________________
    pool1_pad (ZeroPadding2D)       (None, 114, 114, 64) 0           conv1_relu[0][0]                 
    __________________________________________________________________________________________________
    pool1_pool (MaxPooling2D)       (None, 56, 56, 64)   0           pool1_pad[0][0]                  
    __________________________________________________________________________________________________
    conv2_block1_1_conv (Conv2D)    (None, 56, 56, 64)   4160        pool1_pool[0][0]                 
    __________________________________________________________________________________________________
    conv2_block1_1_bn (BatchNormali (None, 56, 56, 64)   256         conv2_block1_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block1_1_relu (Activation (None, 56, 56, 64)   0           conv2_block1_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv2_block1_2_conv (Conv2D)    (None, 56, 56, 64)   36928       conv2_block1_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv2_block1_2_bn (BatchNormali (None, 56, 56, 64)   256         conv2_block1_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block1_2_relu (Activation (None, 56, 56, 64)   0           conv2_block1_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv2_block1_0_conv (Conv2D)    (None, 56, 56, 256)  16640       pool1_pool[0][0]                 
    __________________________________________________________________________________________________
    conv2_block1_3_conv (Conv2D)    (None, 56, 56, 256)  16640       conv2_block1_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv2_block1_0_bn (BatchNormali (None, 56, 56, 256)  1024        conv2_block1_0_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block1_3_bn (BatchNormali (None, 56, 56, 256)  1024        conv2_block1_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block1_add (Add)          (None, 56, 56, 256)  0           conv2_block1_0_bn[0][0]          
                                                                     conv2_block1_3_bn[0][0]          
    __________________________________________________________________________________________________
    conv2_block1_out (Activation)   (None, 56, 56, 256)  0           conv2_block1_add[0][0]           
    __________________________________________________________________________________________________
    conv2_block2_1_conv (Conv2D)    (None, 56, 56, 64)   16448       conv2_block1_out[0][0]           
    __________________________________________________________________________________________________
    conv2_block2_1_bn (BatchNormali (None, 56, 56, 64)   256         conv2_block2_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block2_1_relu (Activation (None, 56, 56, 64)   0           conv2_block2_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv2_block2_2_conv (Conv2D)    (None, 56, 56, 64)   36928       conv2_block2_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv2_block2_2_bn (BatchNormali (None, 56, 56, 64)   256         conv2_block2_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block2_2_relu (Activation (None, 56, 56, 64)   0           conv2_block2_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv2_block2_3_conv (Conv2D)    (None, 56, 56, 256)  16640       conv2_block2_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv2_block2_3_bn (BatchNormali (None, 56, 56, 256)  1024        conv2_block2_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block2_add (Add)          (None, 56, 56, 256)  0           conv2_block1_out[0][0]           
                                                                     conv2_block2_3_bn[0][0]          
    __________________________________________________________________________________________________
    conv2_block2_out (Activation)   (None, 56, 56, 256)  0           conv2_block2_add[0][0]           
    __________________________________________________________________________________________________
    conv2_block3_1_conv (Conv2D)    (None, 56, 56, 64)   16448       conv2_block2_out[0][0]           
    __________________________________________________________________________________________________
    conv2_block3_1_bn (BatchNormali (None, 56, 56, 64)   256         conv2_block3_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block3_1_relu (Activation (None, 56, 56, 64)   0           conv2_block3_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv2_block3_2_conv (Conv2D)    (None, 56, 56, 64)   36928       conv2_block3_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv2_block3_2_bn (BatchNormali (None, 56, 56, 64)   256         conv2_block3_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block3_2_relu (Activation (None, 56, 56, 64)   0           conv2_block3_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv2_block3_3_conv (Conv2D)    (None, 56, 56, 256)  16640       conv2_block3_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv2_block3_3_bn (BatchNormali (None, 56, 56, 256)  1024        conv2_block3_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block3_add (Add)          (None, 56, 56, 256)  0           conv2_block2_out[0][0]           
                                                                     conv2_block3_3_bn[0][0]          
    __________________________________________________________________________________________________
    conv2_block3_out (Activation)   (None, 56, 56, 256)  0           conv2_block3_add[0][0]           
    __________________________________________________________________________________________________
    conv3_block1_1_conv (Conv2D)    (None, 28, 28, 128)  32896       conv2_block3_out[0][0]           
    __________________________________________________________________________________________________
    conv3_block1_1_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block1_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block1_1_relu (Activation (None, 28, 28, 128)  0           conv3_block1_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block1_2_conv (Conv2D)    (None, 28, 28, 128)  147584      conv3_block1_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv3_block1_2_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block1_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block1_2_relu (Activation (None, 28, 28, 128)  0           conv3_block1_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block1_0_conv (Conv2D)    (None, 28, 28, 512)  131584      conv2_block3_out[0][0]           
    __________________________________________________________________________________________________
    conv3_block1_3_conv (Conv2D)    (None, 28, 28, 512)  66048       conv3_block1_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv3_block1_0_bn (BatchNormali (None, 28, 28, 512)  2048        conv3_block1_0_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block1_3_bn (BatchNormali (None, 28, 28, 512)  2048        conv3_block1_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block1_add (Add)          (None, 28, 28, 512)  0           conv3_block1_0_bn[0][0]          
                                                                     conv3_block1_3_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block1_out (Activation)   (None, 28, 28, 512)  0           conv3_block1_add[0][0]           
    __________________________________________________________________________________________________
    conv3_block2_1_conv (Conv2D)    (None, 28, 28, 128)  65664       conv3_block1_out[0][0]           
    __________________________________________________________________________________________________
    conv3_block2_1_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block2_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block2_1_relu (Activation (None, 28, 28, 128)  0           conv3_block2_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block2_2_conv (Conv2D)    (None, 28, 28, 128)  147584      conv3_block2_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv3_block2_2_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block2_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block2_2_relu (Activation (None, 28, 28, 128)  0           conv3_block2_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block2_3_conv (Conv2D)    (None, 28, 28, 512)  66048       conv3_block2_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv3_block2_3_bn (BatchNormali (None, 28, 28, 512)  2048        conv3_block2_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block2_add (Add)          (None, 28, 28, 512)  0           conv3_block1_out[0][0]           
                                                                     conv3_block2_3_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block2_out (Activation)   (None, 28, 28, 512)  0           conv3_block2_add[0][0]           
    __________________________________________________________________________________________________
    conv3_block3_1_conv (Conv2D)    (None, 28, 28, 128)  65664       conv3_block2_out[0][0]           
    __________________________________________________________________________________________________
    conv3_block3_1_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block3_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block3_1_relu (Activation (None, 28, 28, 128)  0           conv3_block3_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block3_2_conv (Conv2D)    (None, 28, 28, 128)  147584      conv3_block3_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv3_block3_2_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block3_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block3_2_relu (Activation (None, 28, 28, 128)  0           conv3_block3_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block3_3_conv (Conv2D)    (None, 28, 28, 512)  66048       conv3_block3_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv3_block3_3_bn (BatchNormali (None, 28, 28, 512)  2048        conv3_block3_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block3_add (Add)          (None, 28, 28, 512)  0           conv3_block2_out[0][0]           
                                                                     conv3_block3_3_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block3_out (Activation)   (None, 28, 28, 512)  0           conv3_block3_add[0][0]           
    __________________________________________________________________________________________________
    conv3_block4_1_conv (Conv2D)    (None, 28, 28, 128)  65664       conv3_block3_out[0][0]           
    __________________________________________________________________________________________________
    conv3_block4_1_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block4_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block4_1_relu (Activation (None, 28, 28, 128)  0           conv3_block4_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block4_2_conv (Conv2D)    (None, 28, 28, 128)  147584      conv3_block4_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv3_block4_2_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block4_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block4_2_relu (Activation (None, 28, 28, 128)  0           conv3_block4_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block4_3_conv (Conv2D)    (None, 28, 28, 512)  66048       conv3_block4_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv3_block4_3_bn (BatchNormali (None, 28, 28, 512)  2048        conv3_block4_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block4_add (Add)          (None, 28, 28, 512)  0           conv3_block3_out[0][0]           
                                                                     conv3_block4_3_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block4_out (Activation)   (None, 28, 28, 512)  0           conv3_block4_add[0][0]           
    __________________________________________________________________________________________________
    conv4_block1_1_conv (Conv2D)    (None, 14, 14, 256)  131328      conv3_block4_out[0][0]           
    __________________________________________________________________________________________________
    conv4_block1_1_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block1_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block1_1_relu (Activation (None, 14, 14, 256)  0           conv4_block1_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block1_2_conv (Conv2D)    (None, 14, 14, 256)  590080      conv4_block1_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block1_2_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block1_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block1_2_relu (Activation (None, 14, 14, 256)  0           conv4_block1_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block1_0_conv (Conv2D)    (None, 14, 14, 1024) 525312      conv3_block4_out[0][0]           
    __________________________________________________________________________________________________
    conv4_block1_3_conv (Conv2D)    (None, 14, 14, 1024) 263168      conv4_block1_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block1_0_bn (BatchNormali (None, 14, 14, 1024) 4096        conv4_block1_0_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block1_3_bn (BatchNormali (None, 14, 14, 1024) 4096        conv4_block1_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block1_add (Add)          (None, 14, 14, 1024) 0           conv4_block1_0_bn[0][0]          
                                                                     conv4_block1_3_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block1_out (Activation)   (None, 14, 14, 1024) 0           conv4_block1_add[0][0]           
    __________________________________________________________________________________________________
    conv4_block2_1_conv (Conv2D)    (None, 14, 14, 256)  262400      conv4_block1_out[0][0]           
    __________________________________________________________________________________________________
    conv4_block2_1_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block2_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block2_1_relu (Activation (None, 14, 14, 256)  0           conv4_block2_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block2_2_conv (Conv2D)    (None, 14, 14, 256)  590080      conv4_block2_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block2_2_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block2_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block2_2_relu (Activation (None, 14, 14, 256)  0           conv4_block2_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block2_3_conv (Conv2D)    (None, 14, 14, 1024) 263168      conv4_block2_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block2_3_bn (BatchNormali (None, 14, 14, 1024) 4096        conv4_block2_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block2_add (Add)          (None, 14, 14, 1024) 0           conv4_block1_out[0][0]           
                                                                     conv4_block2_3_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block2_out (Activation)   (None, 14, 14, 1024) 0           conv4_block2_add[0][0]           
    __________________________________________________________________________________________________
    conv4_block3_1_conv (Conv2D)    (None, 14, 14, 256)  262400      conv4_block2_out[0][0]           
    __________________________________________________________________________________________________
    conv4_block3_1_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block3_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block3_1_relu (Activation (None, 14, 14, 256)  0           conv4_block3_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block3_2_conv (Conv2D)    (None, 14, 14, 256)  590080      conv4_block3_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block3_2_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block3_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block3_2_relu (Activation (None, 14, 14, 256)  0           conv4_block3_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block3_3_conv (Conv2D)    (None, 14, 14, 1024) 263168      conv4_block3_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block3_3_bn (BatchNormali (None, 14, 14, 1024) 4096        conv4_block3_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block3_add (Add)          (None, 14, 14, 1024) 0           conv4_block2_out[0][0]           
                                                                     conv4_block3_3_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block3_out (Activation)   (None, 14, 14, 1024) 0           conv4_block3_add[0][0]           
    __________________________________________________________________________________________________
    conv4_block4_1_conv (Conv2D)    (None, 14, 14, 256)  262400      conv4_block3_out[0][0]           
    __________________________________________________________________________________________________
    conv4_block4_1_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block4_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block4_1_relu (Activation (None, 14, 14, 256)  0           conv4_block4_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block4_2_conv (Conv2D)    (None, 14, 14, 256)  590080      conv4_block4_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block4_2_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block4_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block4_2_relu (Activation (None, 14, 14, 256)  0           conv4_block4_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block4_3_conv (Conv2D)    (None, 14, 14, 1024) 263168      conv4_block4_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block4_3_bn (BatchNormali (None, 14, 14, 1024) 4096        conv4_block4_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block4_add (Add)          (None, 14, 14, 1024) 0           conv4_block3_out[0][0]           
                                                                     conv4_block4_3_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block4_out (Activation)   (None, 14, 14, 1024) 0           conv4_block4_add[0][0]           
    __________________________________________________________________________________________________
    conv4_block5_1_conv (Conv2D)    (None, 14, 14, 256)  262400      conv4_block4_out[0][0]           
    __________________________________________________________________________________________________
    conv4_block5_1_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block5_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block5_1_relu (Activation (None, 14, 14, 256)  0           conv4_block5_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block5_2_conv (Conv2D)    (None, 14, 14, 256)  590080      conv4_block5_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block5_2_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block5_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block5_2_relu (Activation (None, 14, 14, 256)  0           conv4_block5_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block5_3_conv (Conv2D)    (None, 14, 14, 1024) 263168      conv4_block5_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block5_3_bn (BatchNormali (None, 14, 14, 1024) 4096        conv4_block5_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block5_add (Add)          (None, 14, 14, 1024) 0           conv4_block4_out[0][0]           
                                                                     conv4_block5_3_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block5_out (Activation)   (None, 14, 14, 1024) 0           conv4_block5_add[0][0]           
    __________________________________________________________________________________________________
    conv4_block6_1_conv (Conv2D)    (None, 14, 14, 256)  262400      conv4_block5_out[0][0]           
    __________________________________________________________________________________________________
    conv4_block6_1_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block6_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block6_1_relu (Activation (None, 14, 14, 256)  0           conv4_block6_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block6_2_conv (Conv2D)    (None, 14, 14, 256)  590080      conv4_block6_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block6_2_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block6_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block6_2_relu (Activation (None, 14, 14, 256)  0           conv4_block6_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block6_3_conv (Conv2D)    (None, 14, 14, 1024) 263168      conv4_block6_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block6_3_bn (BatchNormali (None, 14, 14, 1024) 4096        conv4_block6_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block6_add (Add)          (None, 14, 14, 1024) 0           conv4_block5_out[0][0]           
                                                                     conv4_block6_3_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block6_out (Activation)   (None, 14, 14, 1024) 0           conv4_block6_add[0][0]           
    __________________________________________________________________________________________________
    conv5_block1_1_conv (Conv2D)    (None, 7, 7, 512)    524800      conv4_block6_out[0][0]           
    __________________________________________________________________________________________________
    conv5_block1_1_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block1_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block1_1_relu (Activation (None, 7, 7, 512)    0           conv5_block1_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv5_block1_2_conv (Conv2D)    (None, 7, 7, 512)    2359808     conv5_block1_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv5_block1_2_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block1_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block1_2_relu (Activation (None, 7, 7, 512)    0           conv5_block1_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv5_block1_0_conv (Conv2D)    (None, 7, 7, 2048)   2099200     conv4_block6_out[0][0]           
    __________________________________________________________________________________________________
    conv5_block1_3_conv (Conv2D)    (None, 7, 7, 2048)   1050624     conv5_block1_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv5_block1_0_bn (BatchNormali (None, 7, 7, 2048)   8192        conv5_block1_0_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block1_3_bn (BatchNormali (None, 7, 7, 2048)   8192        conv5_block1_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block1_add (Add)          (None, 7, 7, 2048)   0           conv5_block1_0_bn[0][0]          
                                                                     conv5_block1_3_bn[0][0]          
    __________________________________________________________________________________________________
    conv5_block1_out (Activation)   (None, 7, 7, 2048)   0           conv5_block1_add[0][0]           
    __________________________________________________________________________________________________
    conv5_block2_1_conv (Conv2D)    (None, 7, 7, 512)    1049088     conv5_block1_out[0][0]           
    __________________________________________________________________________________________________
    conv5_block2_1_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block2_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block2_1_relu (Activation (None, 7, 7, 512)    0           conv5_block2_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv5_block2_2_conv (Conv2D)    (None, 7, 7, 512)    2359808     conv5_block2_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv5_block2_2_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block2_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block2_2_relu (Activation (None, 7, 7, 512)    0           conv5_block2_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv5_block2_3_conv (Conv2D)    (None, 7, 7, 2048)   1050624     conv5_block2_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv5_block2_3_bn (BatchNormali (None, 7, 7, 2048)   8192        conv5_block2_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block2_add (Add)          (None, 7, 7, 2048)   0           conv5_block1_out[0][0]           
                                                                     conv5_block2_3_bn[0][0]          
    __________________________________________________________________________________________________
    conv5_block2_out (Activation)   (None, 7, 7, 2048)   0           conv5_block2_add[0][0]           
    __________________________________________________________________________________________________
    conv5_block3_1_conv (Conv2D)    (None, 7, 7, 512)    1049088     conv5_block2_out[0][0]           
    __________________________________________________________________________________________________
    conv5_block3_1_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block3_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block3_1_relu (Activation (None, 7, 7, 512)    0           conv5_block3_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv5_block3_2_conv (Conv2D)    (None, 7, 7, 512)    2359808     conv5_block3_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv5_block3_2_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block3_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block3_2_relu (Activation (None, 7, 7, 512)    0           conv5_block3_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv5_block3_3_conv (Conv2D)    (None, 7, 7, 2048)   1050624     conv5_block3_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv5_block3_3_bn (BatchNormali (None, 7, 7, 2048)   8192        conv5_block3_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block3_add (Add)          (None, 7, 7, 2048)   0           conv5_block2_out[0][0]           
                                                                     conv5_block3_3_bn[0][0]          
    __________________________________________________________________________________________________
    conv5_block3_out (Activation)   (None, 7, 7, 2048)   0           conv5_block3_add[0][0]           
    __________________________________________________________________________________________________
    max_pooling2d (MaxPooling2D)    (None, 2, 2, 2048)   0           conv5_block3_out[0][0]           
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 2, 2, 512)    1049088     max_pooling2d[0][0]              
    __________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)  (None, 1, 1, 512)    0           dense[0][0]                      
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 1, 1, 200)    102600      max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 1, 1, 100)    20100       dense_1[0][0]                    
    __________________________________________________________________________________________________
    flatten (Flatten)               (None, 100)          0           dense_2[0][0]                    
    __________________________________________________________________________________________________
    dropout (Dropout)               (None, 100)          0           flatten[0][0]                    
    __________________________________________________________________________________________________
    dense_3 (Dense)                 (None, 3)            303         dropout[0][0]                    
    ==================================================================================================
    Total params: 24,759,803
    Trainable params: 1,172,091
    Non-trainable params: 23,587,712
    __________________________________________________________________________________________________
    


```python
model.compile(loss="categorical_crossentropy",optimizer=Optimizer.Adam(learning_rate=0.001),metrics=['accuracy'])
```


```python
model.fit(train_generator,
          steps_per_epoch=1475//32, 
          validation_data=valid_generator,
          validation_steps=326//32,
          epochs=20)
```

    Epoch 1/20
    46/46 [==============================] - 159s 3s/step - loss: 1.8782 - accuracy: 0.3959 - val_loss: 0.9724 - val_accuracy: 0.4969
    Epoch 2/20
    46/46 [==============================] - 162s 4s/step - loss: 1.0403 - accuracy: 0.5166 - val_loss: 1.0258 - val_accuracy: 0.5813
    Epoch 3/20
    46/46 [==============================] - 166s 4s/step - loss: 1.0150 - accuracy: 0.5026 - val_loss: 0.8983 - val_accuracy: 0.5000
    Epoch 4/20
    46/46 [==============================] - 173s 4s/step - loss: 1.0256 - accuracy: 0.5027 - val_loss: 0.8644 - val_accuracy: 0.5750
    Epoch 5/20
    46/46 [==============================] - 176s 4s/step - loss: 0.9608 - accuracy: 0.5404 - val_loss: 0.8269 - val_accuracy: 0.6094
    Epoch 6/20
    46/46 [==============================] - 172s 4s/step - loss: 0.9271 - accuracy: 0.5873 - val_loss: 0.8475 - val_accuracy: 0.5906
    Epoch 7/20
    46/46 [==============================] - 172s 4s/step - loss: 0.9232 - accuracy: 0.5953 - val_loss: 0.8523 - val_accuracy: 0.6062
    Epoch 8/20
    46/46 [==============================] - 174s 4s/step - loss: 0.9170 - accuracy: 0.6054 - val_loss: 0.8017 - val_accuracy: 0.6094
    Epoch 9/20
    46/46 [==============================] - 173s 4s/step - loss: 0.9188 - accuracy: 0.5929 - val_loss: 0.8224 - val_accuracy: 0.6094
    Epoch 10/20
    46/46 [==============================] - 173s 4s/step - loss: 0.8922 - accuracy: 0.5894 - val_loss: 0.7873 - val_accuracy: 0.6250
    Epoch 11/20
    46/46 [==============================] - 179s 4s/step - loss: 0.9007 - accuracy: 0.5773 - val_loss: 0.8037 - val_accuracy: 0.5875
    Epoch 12/20
    46/46 [==============================] - 176s 4s/step - loss: 0.9009 - accuracy: 0.6003 - val_loss: 0.7482 - val_accuracy: 0.6344
    Epoch 13/20
    46/46 [==============================] - 200s 4s/step - loss: 0.8480 - accuracy: 0.6379 - val_loss: 0.7386 - val_accuracy: 0.6531
    Epoch 14/20
    46/46 [==============================] - 202s 4s/step - loss: 0.8281 - accuracy: 0.6451 - val_loss: 0.8040 - val_accuracy: 0.5969
    Epoch 15/20
    46/46 [==============================] - 184s 4s/step - loss: 0.8878 - accuracy: 0.5930 - val_loss: 0.7944 - val_accuracy: 0.6125
    Epoch 16/20
    46/46 [==============================] - 175s 4s/step - loss: 0.8630 - accuracy: 0.6233 - val_loss: 0.6784 - val_accuracy: 0.7031
    Epoch 17/20
    46/46 [==============================] - 187s 4s/step - loss: 0.8647 - accuracy: 0.6367 - val_loss: 0.6692 - val_accuracy: 0.6844
    Epoch 18/20
    46/46 [==============================] - 190s 4s/step - loss: 0.8331 - accuracy: 0.6079 - val_loss: 0.7793 - val_accuracy: 0.6250
    Epoch 19/20
    46/46 [==============================] - 178s 4s/step - loss: 0.8146 - accuracy: 0.6467 - val_loss: 0.9050 - val_accuracy: 0.5781
    Epoch 20/20
    46/46 [==============================] - 181s 4s/step - loss: 0.8337 - accuracy: 0.6254 - val_loss: 0.7605 - val_accuracy: 0.6344
    




    <tensorflow.python.keras.callbacks.History at 0x27c23b42848>




```python
testAugment=ImageDataGenerator(rescale=1./255)
```


```python
test_generator=testAugment.flow_from_directory(
    testPathmode,
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=1,
    shuffle=False
    )
```

    Found 227 images belonging to 3 classes.
    


```python
test_loss,test_acc=model.evaluate(test_generator)
```

    227/227 [==============================] - 31s 135ms/step - loss: 0.7369 - accuracy: 0.6916
    


```python
print("Test Accuracy: ", test_acc)
```

    Test Accuracy:  0.691629946231842
    


```python

```
