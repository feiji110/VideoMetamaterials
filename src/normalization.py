# 定义了一个名为Normalization的类，用于数据归一化操作。它提供了归一化和反归一化的方法，支持不同的归一化策略和数据类型
import torch
import torch.nn.functional as F
import numpy as np

class Normalization:
    def __init__(self,data,dataType,strategy):
        """_summary_
        Args:
            data (_type_): 要进行归一化的数据
            dataType (_type_): 一个列表,指示每个特征的数据类型（连续或分类）
            strategy (_type_): 归一化策略
        """
        # 计算每一列的均值、标准差、最大值、最小值
        self.mu = torch.mean(data,dim=0)
        self.std = torch.std(data,dim=0)
        self.min = torch.min(data,dim=0)[0] # torch.min返回一个元组，第一个元素为最小值，第二个元素为最小值的下标
        self.max = torch.max(data,dim=0)[0] # torch.max返回一个元组，第一个元素为最大值，第二个元素为最大值的下标
        # 全局最大最小值
        self.globalmin = torch.min(data)
        self.globalmax = torch.max(data)

        self.dataType = dataType   # 数据类型列表
        self.cols = data.size()[1] # 特征的数量
        self.strategy = strategy   # 归一化策略
    
    def normalize(self, data):
        list_index_cat = []       
        temp = torch.zeros(data.shape,device=data.device) # 生成一个与data形状相同的全零张量
        for i in range(0, self.cols): # 遍历每一列特征
            if self.dataType[i] == 'continuous': # 如果特征的数据类型是连续型，则根据归一化策略对数据进行缩放

                if(self.strategy == 'min-max-1'):
                    #scale to [0,1]
                    temp[:,i] = torch.div(data[:,i]-self.min[i], self.max[i]-self.min[i])

                elif(self.strategy == 'global-min-max-1'):
                    #scale to [-1,1] based on min max of full dataset
                    temp[:,i] = torch.div(data[:,i]-self.globalmin, self.globalmax-self.globalmin)

                elif(self.strategy == 'min-max-2'):
                    #scale to [-1,1]
                    temp[:,i] = 2.*torch.div(data[:,i]-self.min[i], self.max[i]-self.min[i])-1.

                elif(self.strategy == 'global-min-max-2'):
                    #scale to [-1,1] based on min max of full dataset
                    temp[:,i] = 2.*torch.div(data[:,i]-self.globalmin, self.globalmax-self.globalmin)-1.

                elif(self.strategy == 'mean-std'):
                    #scale s.t. mean=0, std=1
                    temp[:,i] = torch.div(data[:,i]-self.mu[i], self.std[i])

                elif (self.strategy == 'none'):
                    temp[:,i] = data[:,i]

                else:
                    raise ValueError('Incorrect normalization strategy')

            elif self.dataType[i] == 'categorical': # 如果特征的数据类型是分类型，则将其转换为二进制编码，并将编码后的特征追加到temp张量的末尾
                #convert categorical features into binaries and append at the end of feature tensor
                temp = torch.cat((temp,F.one_hot(data[:,i].to(torch.int64))),dim=1)# F.one_hot()方法将分类特征转换为二进制编码，然后将编码后的特征追加到temp张量的末尾
                list_index_cat = np.append(list_index_cat,i)# 记录分类特征的下标
                                   
            else:
                raise ValueError("Data type must be either continuous or categorical")

        # delete original (not one-hot encoded) categorical features； normalize方法删除了原始的分类特征，只保留了二进制编码后的特征
        j = 0
        for i in np.array(list_index_cat, dtype=np.int64):  # 遍历分类特征列表的下标        
            temp = torch.cat([temp[:,0:i+j], temp[:,i+1+j:]],dim=1)# 删除原始的分类特征
            j -= 1 # 删除一个分类特征后，分类特征列表的下标需要减1

        return temp

    def unnormalize(self, data):
        temp = torch.zeros(data.shape,device=data.device)
        for i in range(0, self.cols):
            if self.dataType[i] == 'continuous':
                
                if(self.strategy == 'min-max-1'):
                    temp[:,i] = torch.mul(data[:,i], self.max[i]-self.min[i]) +self.min[i]

                elif(self.strategy == 'global-min-max-1'):
                    temp[:,i] = torch.mul(data[:,i], self.globalmax-self.globalmin) +self.globalmin

                elif(self.strategy == 'min-max-2'):
                    temp[:,i] = torch.mul(0.5*data[:,i]+0.5, self.max[i]-self.min[i]) +self.min[i]

                elif(self.strategy == 'global-min-max-2'):
                    temp[:,i] = torch.mul(0.5*data[:,i]+0.5, self.globalmax-self.globalmin) +self.globalmin
            
                elif(self.strategy == 'mean-std'):
                    temp[:,i] = torch.mul(data[:,i], self.std[i]) + self.mu[i]

                elif (self.strategy == 'none'):
                    temp[:,i] = data[:,i]

                else:
                    raise ValueError('Incorrect normalization strategy')
                
            elif self.dataType[i] == 'categorical':
                temp[:,i] = data[:,i]

            else:
                raise ValueError("Data type must be either continuous or categorical")
        return temp