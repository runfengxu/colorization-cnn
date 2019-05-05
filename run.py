import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import os
import time


def element_wise_op(array,array2,filter_index):
    [depth,rows,cols]=array.shape
    for i in range(rows):
        for j in range(cols):
            array2[filter_index][i][j]=1.0 / (1.0 + np.exp(-array[filter_index][i][j]))
def get_patch(input_array,i,j,filter_width,filter_height,stride):
    start_i=i*stride
    start_j=j*stride
    
    if input_array.ndim==2:
        input_array_conv=input_array[start_i:start_i+filter_height,start_j:start_j+filter_width]
        
    return input_array_conv


def loss(output_array,output_array_height,output_array_width,standard_array):
    loss=0
    for i in range(output_array_height):
        for j in range(output_array_width):
            loss+=(output_array[i][j]-standard_array[i][j])*(output_array[i][j]-standard_array[i][j])
    
    return loss
def conv(input_array, kernel_array,output_array, stride=1, bias=0):
   

    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (get_patch(input_array, i, j, kernel_width, kernel_height, stride) * kernel_array).sum() + bias

def padding(input_array, zp=1):
   
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((input_depth, input_height + 2 * zp,input_width + 2 * zp))
            padded_array[:,zp : zp + input_height,zp : zp + input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((input_height + 2 * zp,input_width + 2 * zp))
            padded_array[zp : zp + input_height,zp : zp + input_width] = input_array
            return padded_array

class Filter(object):
    def __init__(self, width, height):
        self.weights = np.random.uniform(-1, 1,(height, width))
        self.bias = 0
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0


    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (
            repr(self.weights), repr(self.bias))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights += learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad

        
class ConvLayer(object):
    def __init__(self, input_width, input_height, filter_width, filter_height, filter_number, zero_padding, stride, learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = int(ConvLayer.calculate_output_size(self.input_width, filter_width, zero_padding,stride))
        self.output_height = int(ConvLayer.calculate_output_size(self.input_height, filter_height, zero_padding,stride))
        self.output_array = np.zeros((self.filter_number, self.output_height, self.output_width))
        self.output_array2= np.zeros((self.filter_number, self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width, filter_height))
        self.learning_rate = learning_rate

    def forward(self, input_array,filter_index):
        '''
        计算卷积层的输出
        输出结果保存在self.output_array
        '''
        self.input_array = input_array
        self.padded_input_array = padding(input_array,self.zero_padding)
        filter = self.filters[filter_index]
        conv(self.padded_input_array, filter.get_weights(), self.output_array[filter_index],self.stride, filter.get_bias())
        element_wise_op(self.output_array, self.output_array2,filter_index)

        
        
     
    
    def backward(self, input_array,filter_index):
        fil=self.filters[filter_index]
        for i in range(self.filter_height):
            for j in range(self.filter_width):
                fil.weights_grad[i][j]=0
                for k in range(self.output_height):
                    for l in range(self.output_width):
                        a= get_patch(self.padded_input_array,k,l,self.filter_width,self.filter_height,stride=1)
                        fil.weights_grad[i][j]+= 2*self.output_array2[filter_index][k][l]*(self.output_array[filter_index][k][l]*(1 - self.output_array[filter_index][k][l]))*a[i][j]
            
        fil.update(self.learning_rate)
    
    
    @staticmethod
    def calculate_output_size(input_size,filter_size,zero_padding,stride):
        return (input_size-filter_size+2*zero_padding)/stride+1
            
            
def prepare_img(img):
    img=np.array(img)
    img=rgb2lab(img)
    a=img[:,:,1]
    b=img[:,:,2]
    gray=img[:,:,0]
    arraylist={'a':[],'b':[]}
    interval=[-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100]
    for i in range(len(interval)-1):
        front=interval[i]
        end=interval[i+1]
        array_a=np.zeros((256,256))
        array_b=np.zeros((256,256))
        [rows, cols] = a.shape
        for row in range(rows):
            for col in range(cols):
                if a[row][col]>front and a[row][col]<=end:
                    array_a[row][col]=1.0
                    
                if b[row][col]>front and b[row][col]<=end:
                    array_b[row][col]=1.0
        arraylist['a'].append(array_a)
        arraylist['b'].append(array_b)
    return arraylist,gray





files=os.listdir("training_set/")
t1=time.time()
cl=ConvLayer(input_width=256, input_height=256, filter_width=3, filter_height=3, filter_number=40, zero_padding=1, stride=1, learning_rate=1)
t2=time.time()

# print(t2-t1)
for i in range(1):
    fd=os.path.join("training_set/",files[i])
    img=Image.open(fd)        
    stand_arraylist,gray=prepare_img(img)

    t3=time.time()
   # print(t3-t2)
    for j in range(20):
        print("j=",j)
        cl.forward(gray,j)
        loss0 = loss(cl.output_array2[j],256,256,stand_arraylist['a'][j])
        print("loss0",loss0)
        cl.backward(input_array=gray,filter_index=j)


        cl.forward(gray,j)

        loss1 = loss(cl.output_array2[j],256,256,stand_arraylist['a'][j])
        print("loss1",loss1)
        print("improve",loss1-loss0)
        k=0
        while (k <20):
            cl.backward(input_array=gray,filter_index=j)
            cl.forward(gray,j)
            loss0=loss1
            loss1=loss(cl.output_array2[j],256,256,stand_arraylist['a'][j])
            print("loss1-loss0",abs(loss1-loss0))
            k=k+1 
            print("k=",k)
    for j in range(20,40):
        cl.forward(gray,j)
        loss0 = loss(cl.output_array2[j],256,256,stand_arraylist['b'][j-20])
        t4=time.time()
        cl.backward(input_array=gray,filter_index=j)
        t5=time.time()
        #print(t5-t4)
        cl.forward(gray,j)

        loss1 = loss(cl.output_array2[j],256,256,stand_arraylist['b'][j-20])
        k=0
        while abs(loss1-loss0)>1e-1 and k <20:
            cl.backward(input_array=gray,filter_index=j)
            cl.forward(gray,j)
            loss0=loss1
            loss1=loss(cl.output_array2[j],256,256,stand_arraylist['b'][j-20])
            print(loss1)
            k=k+1 
            print(k)



    
    
    
    
