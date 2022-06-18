import numpy as np
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Lambda, MaxPooling2D, Input,Conv2DTranspose,concatenate
from urllib.request import urlretrieve
import cv2
import h5py
import os, sys
ReloadModel=1
TrainModel=0

ModelName='Model.h5'
TrainData='traindata.h5'
TestData='testdata.h5'


def check_get(url,File_Name): 
    def download_callback(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            s = "\r%5.1f%% %*d MB / %d MB" % (
                percent, len(str(totalsize)), readsofar/1e6, totalsize/1e6)
            sys.stderr.write(s)
            if readsofar >= totalsize: # near the end
                sys.stderr.write("\n")
        else: # total size is unknown
            sys.stderr.write("read %d\n" % (readsofar,))
    if not os.path.isfile(File_Name):
        ans=input('You dont have the file "' +File_Name +'". Do you want to download it? (Y/N) ')    
        if ans=='Y' or ans=='y' or ans=='yes' or ans=='Yes' or ans=='YES':
            print('Beginning file download. This might take several minutes.')
            urlretrieve(url,File_Name,download_callback)
    else:
        print('File "' +File_Name +'" is detected on your machine.'  )
        
def unet(INPUT_SHAPE,OUTPUT_SHAPE):
    inputs = Input(INPUT_SHAPE[1:])
    s = Lambda(lambda x: x ) (inputs)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)
    outputs = Conv2D(OUTPUT_SHAPE[-1], (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse','accuracy']) 
    model.summary()
    return model
def imwr(img,path):
    if np.max(img)<=1:
        img=img*255
    np.squeeze(np.uint8(img))
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path,img)  
def h5size(Name,Field):
    with h5py.File(Name,'r') as f:
        Shape=f[Field].shape  
    return Shape  
def readh5slice(FileName,FieldName,Slices):
    with h5py.File(FileName, "r") as f:
         A=f[FieldName][np.sort(Slices),...]
    return A
def normal(A):
    A_min = np.min(A)
    return (A-A_min)/(np.max(A)-A_min)
def mkdir(Path):
    try:
        os.mkdir(Path)
        print('path created.')
    except:
        print('path already exist.')
check_get('https://zenodo.org/record/6659509/files/traindata.h5?download=1',TrainData)
check_get('https://zenodo.org/record/6659509/files/testdata.h5?download=1',TestData)
INPUT_SHAPE=h5size(TrainData,'X')
OUTPUT_SHAPE=h5size(TrainData,'Y')
if ReloadModel:
    model=keras.models.load_model(ModelName)
else:
    model=unet(INPUT_SHAPE,OUTPUT_SHAPE)
bite=200
if TrainModel:
    for I in range(1):
        A=readh5slice(TrainData,'X',np.arange(int(I*bite),int((I+1)*bite)))
        B=readh5slice(TrainData,'Y',np.arange(int(I*bite),int((I+1)*bite)))
        X=np.float32(A/255)
        Y=np.float32(B/255)
        model.fit(X,Y,batch_size=2,epochs=100,shuffle=True)
    model.save(ModelName)
INPUT_SHAPE=h5size(TestData,'X')
List=[0,1,2,3]
A=readh5slice(TestData,'X',np.array(List))
B=readh5slice(TestData,'Y',np.array(List))
C=readh5slice(TestData,'Y0',np.array(List))
X=np.float32(A/255)
Y=np.float32(B/255)
Y0=np.float32(C/255)
Y2=model.predict(X)
Y3=np.float32(Y2)
X3=np.float32(X)
Y3=Y3*255
X3=X3*255
Y3=Y3-127
Y3=Y3/2
X3=X3-Y3
X3=normal(X3)
for samp in range(4):
    mkdir('Output/'+str(samp))
    imwr(X[samp,:,:,:],'Output/'+str(samp)+'/'+'low-res.png')
    imwr(Y[samp,:,:,:],'Output/'+str(samp)+'/'+'residual-truth.png')
    imwr(Y2[samp,:,:,:],'Output/'+str(samp)+'/'+'residual-predicted.png')
    imwr(Y0[samp,:,:,:],'Output/'+str(samp)+'/'+'high-res-truth.png')
    imwr(X3[samp,:,:,:],'Output/'+str(samp)+'/'+'high-res-predicted.png')
print('Predictions are saved in the Output folder.')