import numpy as np

class CSK:
    def __init__(self):
        self.eta = 0.075
        self.sigma = 0.2
        self.lmbda = 0.01

    def init(self,frame,x1,y1,width,height):
        # Save position and size of bbox
        self.x1 = x1
        self.y1 = y1
        self.width = width if width%2==0 else width-1
        self.height = height if height%2==0 else height-1
        
        # Crop & Window
        self.x = self.crop(frame,x1,y1,self.width,self.height)
        
        # Generate regression target
        self.y = self.target(self.width,self.height)
        #总的来说，这句话的意思就是找到y中最大值的索引，argmax()找到的是多维数组一维化后最大元素对应的索引，unravel_index()将这个一维的索引还原到多维的索引
        self.prev = np.unravel_index(np.argmax(self.y, axis=None), self.y.shape) # Maximum position

        # Training
        self.alphaf = self.training(self.x,self.y,self.sigma,self.lmbda)

    def update(self,frame):
        # Crop at the previous position (doubled size)
        #在新图片上的目标之前的位置上截取一个两倍长宽大小的候选区域
        z = self.crop(frame,self.x1,self.y1,self.width,self.height)

        # Detection
        #在该区域内寻找目标（用滤波器求最大响应）
        responses = self.detection(self.alphaf,self.x,z,0.2)
        curr = np.unravel_index(np.argmax(responses, axis=None), responses.shape)
        #当前目标中心位置与之前目标中心位置做差
        dy = curr[0]-self.prev[0]
        dx = curr[1]-self.prev[1]

        # New position (left top corner)
        #更新目标Bounding box，为什么是-？
        self.x1 = self.x1 - dx
        self.y1 = self.y1 - dy

        # Training
        #使用新的结果更新模型,a存储中间结果
        a = self.crop(frame,self.x1,self.y1,self.width,self.height)
        xtemp = self.eta*a + (1-self.eta)*self.x
        self.x = a

        self.alphaf = self.eta*self.training(self.x,self.y,0.2,0.01) + (1-self.eta)*self.alphaf # linearly interpolated
        self.x = xtemp

        return self.x1, self.y1

    #求解高斯核函数
    def dgk(self, x1, x2, sigma):
        #conj()函数：求复共轭，fft2():二维傅里叶变换，ifft2():二维傅里叶逆变换，fftshift():将零频率分量移到频谱的中心。
        c = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(x1)*np.conj(np.fft.fft2(x2))))
        #flatten():将数组变成一维
        d = np.dot(np.conj(x1.flatten(1)),x1.flatten(1)) + np.dot(np.conj(x2.flatten(1)),x2.flatten(1)) - 2*c
        #为什么要除以np.size(x1)？
        k = np.exp(-1/sigma**2*np.abs(d)/np.size(x1))
        return k

    def training(self, x, y, sigma, lmbda):
        k = self.dgk(x, x, sigma)
        alphaf = np.fft.fft2(y)/(np.fft.fft2(k)+lmbda)
        #返回的是经过傅里叶变化的alpha，或者说返回的alpha未经过理论公式中的傅里叶逆变换
        return alphaf

    def detection(self, alphaf, x, z, sigma):
        k = self.dgk(x, z, sigma)
        #对应论文中公式9，求最大响应
        responses = np.real(np.fft.ifft2(alphaf*np.fft.fft2(k)))
        return responses

    def window(self,img):
        height = img.shape[0]
        width = img.shape[1]
        print(height)
        print(width)
        j = np.arange(0,width)
        i = np.arange(0,height)
        J, I = np.meshgrid(j,i)
        #对应论文4.1节，消除边界的不连续
        window = np.sin(np.pi*J/width)*np.sin(np.pi*I/height)
        #除以255应该是为了归一化，-0.5是为了做到均值为0，在别处看的，不知道对不对？这样做的原因是什么？
        windowed = window*((img/255)-0.5)
        return windowed

    def crop(self,img,x1,y1,width,height):
        pad_y = [0,0]
        pad_x = [0,0]

        if (y1-height/2) < 0:
            y_up = 0
            pad_y[0] = int(-(y1-height/2))
        else:
            y_up = int(y1-height/2)

        if (y1+3*height/2) > img.shape[0]:
            y_down = img.shape[0]
            pad_y[1] = int((y1+3*height/2) - img.shape[0])
        else:
            y_down = int(y1+3*height/2)

        if (x1-width/2) < 0:
            x_left = 0
            pad_x[0] = int(-(x1-width/2))
        else:
            x_left = int(x1-width/2)

        if (x1+3*width/2) > img.shape[1]:
            x_right = img.shape[1]
            pad_x[1] = int((x1+3*width/2) - img.shape[1])
        else:
            x_right = int(x1+3*width/2)
        #可能是截取感兴趣区域，横向左右扩展半个宽度，纵向上下扩展半个宽度
        cropped = img[y_up:y_down,x_left:x_right]
        #如果在原图片上不能上下扩展半个高度/左右不能扩展半个宽度，则用边界值进行填充，已达扩展到足够大小
        padded = np.pad(cropped,(pad_y,pad_x),'constant',constant_values=(122,122))
        windowed = self.window(padded)
        return windowed

    #对应论文中的4.2部分
    #下面的注释是我的个人理解
    def target(self,width,height):
        #选取感兴趣区域的时候进行了扩展，即长宽各变为原来的两倍
        double_height = 2 * height
        double_width = 2 * width
        #后面为什么要除以这个数？不知道
        s = np.sqrt(double_height*double_width)/16

        j = np.arange(0,double_width)
        i = np.arange(0,double_height)
        J, I = np.meshgrid(j,i)
        #(J-width)**2+(I-height)**2可以理解为距离目标中心点的距离的平方，因为进行了扩展，所以(width,height)可以表示目标中心点坐标
        y = np.exp(-((J-width)**2+(I-height)**2)/s**2)
        return y
