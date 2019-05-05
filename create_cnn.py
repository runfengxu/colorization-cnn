import numpy as np

class ReluActivator(object):
    def forward(self, weighted_input):
        #return weighted_input
        return max(0, weighted_input)

    def backward(self, output):
        return 1 if output > 0 else 0
    
class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1

    
class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))
    #the partial of sigmoid
    def backward(self, output):
        return output * (1 - output)
    
def element_wise_op(array,array2):
    [rows,cols]=array.shape
    for i in range(rows):
        for j in range(cols):
            array2[i][j]=1.0 / (1.0 + np.exp(-array1[i][j]))
    
def get_patch(input_array,i,j,filter_width,filter_height,stride):
    start_i=i*stride
    start_j=j*stride
    
    if input_array.ndim==2:
        input_array_conv=input_array[start_i:start:i+filter_height,start_j:start_j+filter_width]
        
    return input_array_conv


def loss(output_array,output_array_height,output_array_width,standard_array):
    loss=0
    for i in range(output_array_height):
        for j in range(ouput_array_width):
            loss+=(ouput_array[i][j]-standard[i][j])*(ouput_array[i][j]-standard[i][j])
    
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
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1, 1,(depth, height, width))
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
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad

        
class ConvLayer(object):
    def __init__(self, input_width, input_height, filter_width, filter_height, filter_number, zero_padding, stride, activator,learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = ConvLayer.calculate_output_size(self.input_width, filter_width, zero_padding,stride)
        self.output_height = ConvLayer.calculate_output_size(self.input_height, filter_height, zero_padding,stride)
        self.output_array = np.zeros((self.filter_number, self.output_height, self.output_width))
        self.output_array2= np.zeros((self.filter_number, self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width, filter_height))
        self.activator = activator
        self.learning_rate = learning_rate

    def forward(self, input_array):
        '''
        计算卷积层的输出
        输出结果保存在self.output_array
        '''
        self.input_array = input_array
        self.padded_input_array = padding(input_array,self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array, filter.get_weights(), self.output_array[f],self.stride, filter.get_bias())
        element_wise_op(self.output_array, self.output_array2)

        
        
     
    
    def backward(self, input_array):
        for fil in self.filters:
            for i in range(self.filter_height):
                for j in range(self.filter_width):
                    fil.weights_grad[i][j]=0
                    for k in range(self.ouput_height):
                        for l in range(self.output_width):
                            a= get_patch(input_array,k,l,self.filter_width,self.filter_height,stride=1)
                            fil.weights_grad[i][j]+=2*self.ouput_array2[k][l]*(self.output_array[k][l] * (1 - self.output_array[i][j]))*a[i][j]
            
            fil.update()
        
        

