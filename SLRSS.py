import scipy.io as sio
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics,preprocessing
import torch.nn as nn
import torch.functional as F
import torch
from tqdm import tqdm
import time
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


nGpus=0
if torch.cuda.device_count()>1:
    nGpus=torch.cuda.device_count()
    print('Use ',nGpus,' GPUs')

ratio = 0.001  # 使用的回归训练集比例
cls_epoch = 115  # CLS的epoch数

root_dir=os.getcwd()
basedir=os.path.join(root_dir,'MT')  # 不同方法的文件夹
# python ./MT/MTnet_2sCE.py --data_name IndianPines --batch_size 1024 --train_size 5 
parser=argparse.ArgumentParser()
parser.add_argument('--device',action='store',type=str,default='cuda:0',required=False,help='cpu|cuda:0')
parser.add_argument('--seed',type=int,default=2024,required=False,help='random seed')
parser.add_argument('--isTest',action='store_true',required=False,help='test or train flag')
# 数据、训练相关参数
parser.add_argument('--Dataset_path',action='store',type=str,default=os.path.join(root_dir,'Datasets'),required=False,help='the datasets path')
parser.add_argument('--result_path',action='store',type=str,default=os.path.join(root_dir,'results'),required=False,help='the datasets path')
parser.add_argument('--data_name',action='store',type=str,default='Salinas',required=False,help='IndianPines|PaviaU|Salinas|Houston2013|LongKou')
parser.add_argument('--runs',type=int,default=5,required=False,help='repeat run several times')
# -1 表示未指定，将在Args初始化时使用默认值
parser.add_argument('--batch_size',type=int,default='-1',required=False,help='batch_size')
parser.add_argument('--patch_size',type=int,default=-1,required=False,help='patch_size')
parser.add_argument('--train_size',type=float,default=-1.0,required=False,help='train_size')
parser.add_argument('--epochs',type=int,default=-1,required=False,help='epochs')
pargs=parser.parse_args()

Dataset_path=pargs.Dataset_path
result_path = pargs.result_path
IndianPine_path=os.path.join(Dataset_path,'IndianPines')
PaviaU_path=os.path.join(Dataset_path,'PaviaU')
Salinas_path=os.path.join(Dataset_path,'Salinas')
Houston2013_path=os.path.join(Dataset_path,'Houston2013')
LongKou_path = os.path.join(Dataset_path, 'LongKou')

seed=pargs.seed
np.random.seed(seed)
torch.manual_seed(seed)
device=pargs.device
if not torch.cuda.is_available():
    device='cpu'

def preprocess(dataset, normalization = 2):
    '''
    对数据集进行归一化；
    normalization = 1 : 0-1归一化；  2：标准化； 3：分层标准化；4.逐点正则化
    '''
    #attation 数据归一化要做在划分训练样本之前；
    dataset = np.array(dataset, dtype = 'float64')
    [m,n,b] = np.shape(dataset)
    #先进行数据标准化再去除背景  效果更好
    if normalization ==1:
        dataset = np.reshape(dataset, [m*n,b])
        min_max_scaler = preprocessing.MinMaxScaler()
        dataset = min_max_scaler.fit_transform(dataset)
        dataset = dataset.reshape([m,n,b])
    elif normalization ==2:#标准化就是将一行(一列)数据减去均值，除以标准差。最后得到的结果是，对每行/每列来说所有数据都聚集在0附近，方差值为1。
        dataset = np.reshape(dataset, [m*n,b,])
        stand_scaler = preprocessing.StandardScaler()
        dataset = stand_scaler.fit_transform(dataset)
        dataset = dataset.reshape([m,n,b])
    elif normalization ==3:
        stand_scaler = preprocessing.StandardScaler()
        for i in range(b):
            dataset[:,:,i] = stand_scaler.fit_transform(dataset[:,:,i])
    elif normalization ==6:
        stand_scaler = preprocessing.MinMaxScaler()
        for i in range(b):
            dataset[:,:,i] = stand_scaler.fit_transform(dataset[:,:,i])
    elif normalization == 4:
        for i in range(m):
            for j in range(n):
                dataset[i,j,:] = preprocessing.normalize(dataset[i,j,:].reshape(1,-1))[0]
    elif normalization == 5:
        stand_scaler = preprocessing.StandardScaler()
        for i in range(m):
            for j in range(n):
                dataset[i,j,:] = stand_scaler.fit_transform(dataset[i,j,:].reshape(-1,1)).flatten()
    elif normalization == 7:
        stand_scaler = preprocessing.MinMaxScaler()
        for i in range(m):
            for j in range(n):
                dataset[i,j,:] = stand_scaler.fit_transform(dataset[i,j,:].reshape(-1,1)).flatten()
    else:
        pass

    return dataset


def my_pca(image,n_components=0.99,**kwargs):
    """对image 的通道 进行pca，并恢复到image"""
    w,h,b=len(image),len(image[0]),len(image[0][0])
    image = np.reshape(image, [w*h, b])
    pca=PCA(n_components=n_components)
    pca.fit(image)
    return np.reshape(pca.transform(image), [w,h,-1])


def my_train_test_split(data,labels,train_rate,random_state):
    trainX, testX, trainY, testY=None,None,None,None
    for it in np.unique(labels):
        if it == 0:
            continue
        it_index=np.argwhere(labels==it)[:,0] #np.argwhere返回非0的数组元组的索引的列表，按类划分训练集、测试集和验证集
        if (train_rate<1.0 and it_index.shape[0]*train_rate<=5): #train_rate<1.0时，训练样本个数为it_index.shape[0]*train_rate，若此时样本总数不足五个
            itrainX, itestX, itrainY, itestY = train_test_split(data[it_index,:], labels[it_index], train_size=5,#则设置训练样本数为5
                                                                random_state=random_state)#随机数种子，应用于分割前对数据的洗牌
            # 设成定值意味着，对于同一个数据集，只有第一次运行是随机的，随后多次分割只要rondom_state相同，则划分结果也相同
        elif (train_rate>1.0 and it_index.shape[0]<=train_rate) :#train_rate>1.0时，训练样本个数为train_rate，若此时样本总数不足train_rate个
            itrainX, itestX, itrainY, itestY = train_test_split(data[it_index, :], labels[it_index], train_size=10,#则设置训练样本数为10
                                                                random_state=random_state)
        else:
            itrainX, itestX, itrainY, itestY= train_test_split(data[it_index,:], labels[it_index], train_size=train_rate,
                                                    random_state=random_state)
        trainX=np.concatenate((trainX,itrainX)) if trainX is not None else itrainX
        trainY=np.concatenate((trainY,itrainY)) if trainY is not None else itrainY
        testX=np.concatenate((testX,itestX)) if testX is not None else itestX
        testY=np.concatenate((testY,itestY)) if testY is not None else itestY
    return trainX, testX, trainY, testY


def print_metrics(y_true,y_pred,show=True):
    print('metrics : -------------------------')
    kp= metrics.cohen_kappa_score(y_true,y_pred)
    classify_report = metrics.classification_report(y_true, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    overall_accuracy = metrics.accuracy_score(y_true, y_pred)
    acc_for_each_class = metrics.recall_score(y_true, y_pred, average=None)
    average_accuracy = np.mean(acc_for_each_class)

    "new  add"
    writelogtxt(args.logFile, 'metrics : -------------------------')
    writelogtxt(args.logFile, 'classify_report : ')
    writelogtxt(args.logFile, str(classify_report))
    writelogtxt(args.logFile, 'confusion_matrix : ')
    writelogtxt(args.logFile, str(confusion_matrix))
    writelogtxt(args.logFile, 'acc_for_each_class : ')
    writelogtxt(args.logFile, str(acc_for_each_class))
    writelogtxt(args.logFile, '-----------------------------------------')
    writelogtxt(args.logFile, "oas: " + str(format(overall_accuracy)))
    writelogtxt(args.logFile, "aas: " + str(format(average_accuracy)))
    writelogtxt(args.logFile, "kappa: " + str(format(kp)))
    "new  add end"

    print('classify_report : \n', classify_report)
    print('confusion_matrix : \n', confusion_matrix)
    print('acc_for_each_class : \n', acc_for_each_class)
    print('-----------------------------------------')
    print('overall_accuracy: {0:f}'.format(overall_accuracy))
    print('average_accuracy: {0:f}'.format(average_accuracy))
    print('kappa:',kp)
    return overall_accuracy,average_accuracy,kp,classify_report,confusion_matrix,acc_for_each_class


def saveFig(img,imgname):
    h,w=img.shape
    plt.imshow(img,'jet')
    plt.axis('off')
    plt.gcf().set_size_inches(w / 100.0 / 3.0, h / 100.0 / 3.0)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig(imgname,dpi=600)


def load_data(data_name,nC=50):
    
    if data_name=='IndianPines':
        image=sio.loadmat(os.path.join(IndianPine_path,'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels=sio.loadmat(os.path.join(IndianPine_path,'labels.mat'))['labels'].reshape(145,145).T    # 特别对待!!
        
    elif data_name=='PaviaU':
        image=sio.loadmat(os.path.join(PaviaU_path,'PaviaU.mat'))['paviaU']
        labels=sio.loadmat(os.path.join(PaviaU_path,'PaviaU_gt.mat'))['paviaU_gt']
        
    elif data_name=='Salinas':
        image = sio.loadmat(os.path.join(Salinas_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(Salinas_path, 'Salinas_gt.mat'))['salinas_gt']

    elif data_name=='Houston2013':
        image = sio.loadmat(os.path.join(Houston2013_path, 'Houston2013.mat'))['Houston']
        labels = sio.loadmat(os.path.join(Houston2013_path, 'Houston2013_gt.mat'))['labels']

    elif data_name == 'LongKou':
        image = sio.loadmat(os.path.join(LongKou_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
        labels = sio.loadmat(os.path.join(LongKou_path, 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']

    edge_label_path=os.path.join(Dataset_path,data_name+'_edgelabels.mat')  # _edgelabels.mat->_edgelabels_a/b.mat
    edge_dic=sio.loadmat(edge_label_path)  # 像素点的右侧、下侧、右下侧和右上侧，边缘不存在的位置为0
    return np.array(image,dtype=np.float64),np.array(labels,dtype=np.int32),edge_dic

class MyData():
    def __init__(self,image,labels,edge_dic,train_ind,test_ind,val_ind=[],patch_size=1,batch_size=128):
        w,h,b=image.shape
        
        self.batch_size=batch_size
        self.patch_size=patch_size
        self.channels=image.shape[-1]
        self.image_shape=image.shape
        self.labels=labels
        self.image_pad=np.zeros((w+patch_size-1,h+patch_size-1,b))
        self.image_pad[patch_size//2:w+patch_size//2,patch_size//2:h+patch_size//2,:]=image  # padding操作
        self.edge_dic=edge_dic
        self.edge_dic_keys=['edgerow','edgecol','edgeabove','edgebelow']
        self.xd=[0,1,-1,1]  # 代表位置关系
        self.yd=[1,0,1,1]
        # 分类训练集
        self.train_ind=train_ind
        self.test_ind=test_ind
        self.val_ind=val_ind  # 数据被划分：训练集+（测试集+验证集）
        self.train_iters=len(self.train_ind)//batch_size+1
        self.test_iters = len(self.test_ind) // batch_size+1
        self.val_iters=len(self.val_ind)//batch_size+1
        self.val_batch=self.batch_size
        if self.val_iters==1: # 若果验证集太少或没有，则规定其为4份，val_batch ！= batch_size
            self.val_iters=4
            self.val_batch=len(self.val_ind)//self.val_iters+1 
        self.all_indices=np.concatenate([self.train_ind,self.val_ind,self.test_ind],axis=0)
        self.all_iters = (len(self.all_indices)) // self.batch_size + 1
        
        np.random.shuffle(self.train_ind)
        # 回归训练集(所有的边,横、竖、斜上、斜下)    
        self.regress_ind=self.get_regress_ind()
        self.regress_ind = self.regress_ind[np.random.choice(self.regress_ind.shape[0], int(self.regress_ind.shape[0]*ratio), replace=False), :]

        np.random.shuffle(self.regress_ind)  # 多维矩阵中，只对第一维（行）做打乱顺序操作，因此每一类边的内部顺序不变，只改变边类别顺序
        self.reg_iters=len(self.regress_ind)//batch_size+1
        if pargs.data_name=='Houston2013':
            np.random.shuffle(self.regress_ind)
            self.reg_iters=int(self.reg_iters/3)
        

    def get_regress_ind(self):
        # 回归训练集(所有的边,横、竖、斜上、斜下)    
        # edgerow=np.zeros((w,h)) # row,col range:[0,w-1],[0,h-2] 可取到 cnt=(w)*(h-1)
        # edgecol=np.zeros((w,h))# row,col range:[0,w-2],[0,h-1]  cnt=(w-1)*(h)
        # edgeabove=np.zeros((w,h))# row,col range:[1,w-1],[0,h-2] cnt=(w-1)*(h-1)
        # edgebelow=np.zeros((w,h))# row,col range:[0,w-2],[0,h-2] cnt=(w-1)*(h-1)
        # 用[a,x,y]表示位置[x,y] Type A的边 
        w,h=self.image_shape[0:2]
        grids=[]
        a,b,c=np.mgrid[0:1,0:w,0:h-1]#返回多维结构，常见的如2D图形，3D图形
        grid=np.c_[a.ravel(),b.ravel(),c.ravel()] #ravel将数组维度拉成一维数组，np.c_连接矩阵
        grids.append(grid)
        a,b,c=np.mgrid[1:2,0:w-1,0:h]
        grid=np.c_[a.ravel(),b.ravel(),c.ravel()]
        grids.append(grid)
        a,b,c=np.mgrid[2:3,1:w,0:h-1]
        grid=np.c_[a.ravel(),b.ravel(),c.ravel()]
        grids.append(grid)
        a,b,c=np.mgrid[3:4,0:w-1,0:h-1]
        grid=np.c_[a.ravel(),b.ravel(),c.ravel()]
        grids.append(grid)
        grids=np.concatenate([grids[0],grids[1],grids[2],grids[3]],axis=0) # 边权图中权重生成矩阵

        # edges_ind=[]
        
        # for i in range(w):
        #     for j in range(h):
        #         if j!=h-1:# edge row
        #             edges_ind.append([0,i,j])
        #         if i!=w-1:# edge col 
        #             edges_ind.append([1,i,j])
        #         if i!=0 and j!=h-1:# edge above
        #             edges_ind.append([2,i,j])
        #         if i!=w-1 and j!=h-1:# edge below
        #             edges_ind.append([3,i,j])
        return grids 


    def get_batch_reg(self,iter):#获取第iter下所有batch回归样本，具体是第iter下第ind_i个batch中，某个边对应两个顶点坐标处在padding后的patch块的图像信息和标签累积

        if iter == self.reg_iters-1:
            indices=self.regress_ind[-self.batch_size:]#regress_ind---->三维(边的类型，边的位置=（图像中的长宽位置）)
        else:
            indices=self.regress_ind[iter*self.batch_size:(iter+1)*self.batch_size]
        hp=self.patch_size
        X=np.zeros((self.batch_size,hp,hp,self.channels),dtype=np.float32)
        X2=np.zeros((self.batch_size,hp,hp,self.channels),dtype=np.float32)
        Yc=np.zeros((self.batch_size,),dtype=np.float32)
        Yc2=np.zeros((self.batch_size,),dtype=np.float32)
        Yr=np.zeros((self.batch_size,1),dtype=np.float32)

        for ind_i, edge_i in enumerate(indices):#indices---->第iter下的batch的数据，四维 (batch_size, (边的类型的位置，边的位置=(图像中的长宽位置)))
            edge_type, edge_pos = self.edge_dic_keys[edge_i[0]], edge_i[1:]
            Yr[ind_i] = self.edge_dic[edge_type][edge_pos[0], edge_pos[1]]#通过边的类型和位置得到边权重字典edge_dic中权重值

            a, b = edge_pos
            X[ind_i,:,:,:] = self.image_pad[a:a+hp,b:b+hp,:]   #ind_i--->batch的批次数，X--->第ind_i批padding后图片的内容
            if [a,b] in self.train_ind:      #对于属于训练集的回归集样本，直接赋予其训练集的真实标签
                Yc[ind_i]=self.labels[a,b]   #Yc--->X对应标签
            else:
                Yc[ind_i]=-1   #-1代表非训练集
            a=edge_pos[0]+self.xd[edge_i[0]]    #xd: list[int] = [0,1,-1,1]
            b=edge_pos[1]+self.yd[edge_i[0]]    #yd: list[int] = [1,0,1,1]  通过横纵坐标加减变化，得到了对应边另一端顶点的位置
            X2[ind_i,:,:,:]=self.image_pad[a:a+hp,b:b+hp,:]
            # set the labels not in trainset to -1 ,discard its real label value 
            if [a,b] in self.train_ind:
                Yc2[ind_i]=self.labels[a,b]
            else:
                Yc2[ind_i]=-1
        return np.array(X,dtype=np.float32),np.array(Yc,dtype=np.float32)-1,\
            np.array(X2,dtype=np.float32),np.array(Yc2,dtype=np.float32)-1,\
            np.array(Yr,dtype=np.float32)
    

    def get_batch(self,iter,testOrval='test'): #获取第iter下（test|val|all|train）中所有batch分类样本，X[ind_i, :, :, :]与Y[ind_i]--->第ind_i批次的样本数据（长宽高数值）与对应标签
        '''test|val|all|train'''
        tmp_batch=self.batch_size
        if testOrval=='test':
            if iter == self.test_iters - 1:
                # indices = self.test_ind[-self.batch_size:]
                indices = self.test_ind[iter * self.batch_size:]
                tmp_batch = len(indices)
            else:
                indices = self.test_ind[iter * self.batch_size:(iter + 1) * self.batch_size]
        elif testOrval=='val':
            tmp_batch = self.val_batch
            if iter == self.val_iters - 1:
                indices = self.val_ind[-self.val_batch:]
                # indices = self.val_ind[iter * self.batch_size:]
                tmp_batch = len(indices)
            else:
                indices = self.val_ind[iter * self.val_batch:(iter + 1) * self.val_batch]
        elif testOrval=='all':
            if iter == self.all_iters - 1:
                # indices = self.all_indices[-self.batch_size:]
                indices = self.all_indices[iter * self.batch_size:]
                tmp_batch=len(indices)
            else:
                indices = self.all_indices[iter * self.batch_size:(iter + 1) * self.batch_size]
        elif testOrval=='train':
            # the val_batch also serves as the tmp_batch of trainset, as train and val are same small size 
            tmp_batch = self.val_batch
            if iter == self.train_iters - 1:
                indices = self.train_ind[-self.val_batch:]
                # indices = self.val_ind[iter * self.batch_size:]
                tmp_batch = len(indices)
            else:
                indices = self.train_ind[iter * self.val_batch:(iter + 1) * self.val_batch]


        X = np.zeros((tmp_batch, self.patch_size, self.patch_size, self.channels))
        Y = np.zeros((tmp_batch,))
        hp = self.patch_size

        for ind_i, ind in enumerate(indices):
            # Pay attention to coordinate changes
            X[ind_i, :, :, :] = self.image_pad[ind[0]:ind[0] + hp, ind[1]:ind[1] + hp, :]
            Y[ind_i] = self.labels[ind[0], ind[1]]
        return np.array(X,dtype=np.float32),np.array(Y,dtype=np.float32)-1,indices


    def shuffle(self):
        np.random.shuffle(self.train_ind) # 打乱了批次的顺序
        np.random.shuffle(self.regress_ind)
        np.random.shuffle(self.val_ind)
        np.random.shuffle(self.all_indices)


class MyNet(nn.Module):
    def __init__(self,out_dim,bands,patch_size=1,batch_size=128):
        super(MyNet,self).__init__()
        self.patch_size=patch_size
        self.batch_size=batch_size
        self.bands=bands
        self.out_dim=out_dim
        self.batch_nGpus=self.batch_size//nGpus if nGpus>1 else self.batch_size
        
        self.conv3dseq=nn.Sequential(
            nn.Conv3d(1,8,kernel_size=(7,3,3),padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=(5,3, 3),padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3),padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )
        
        self.f_shape1=self.get_fea_nums()# (batch,channel,w,h)
        self.conv2d1=nn.Sequential(
            nn.Conv2d(self.f_shape1[1],64,kernel_size=3,padding=1), #f_shape1[1]变形后数据的通道数
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        fc_in=64*self.f_shape1[2]*self.f_shape1[3] #f_shape1[2]， f_shape1[3]变形后数据的长宽，等于输入三维卷积模块前的长宽大小
        self.fcseq=nn.Sequential(
            nn.Linear(fc_in,256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,self.out_dim)
        )
        self.fcreg=nn.Sequential(
            nn.Linear(fc_in,256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,1)
        )


    def forward(self,X1,X2,isTest=False):
        # isTest flag just indicates the single branch net ,maybe not for test.
        if isTest:
            X1=self.conv3dseq(X1)
            X1=X1.view(-1,self.f_shape1[1],self.f_shape1[2],self.f_shape1[3])
            X1=self.conv2d1(X1)
            X1 = X1.view(X1.shape[0], -1)
            X1=self.fcseq(X1)
            return X1

        X1=self.conv3dseq(X1)
        X1=X1.view(-1,self.f_shape1[1],self.f_shape1[2],self.f_shape1[3])
        X1=self.conv2d1(X1)
        X1 = X1.view(self.batch_nGpus, -1)

        X2=self.conv3dseq(X2)
        X2=X2.view(-1,self.f_shape1[1],self.f_shape1[2],self.f_shape1[3])
        X2=self.conv2d1(X2)
        X2 = X2.view(self.batch_nGpus, -1)

        Xr=X1+X2
        Xr=self.fcreg(Xr)
        X1=self.fcseq(X1)
        X2=self.fcseq(X2)
        
        return X1,X2,Xr


    def get_fea_nums(self):  #一个通道的数据三维卷积模块后得到的数据，经过变形成原来输入长宽大小后的结果
        x=torch.ones(self.batch_size,1,self.bands,self.patch_size,self.patch_size)
        x=self.conv3dseq(x)
        _,_,_,win_w,win_h=x.shape
        a=x.view([self.batch_size,-1,win_w,win_h]).shape
        return a
    

    def test_forward(self,X1): #测试和分类模块
        X1=self.conv3dseq(X1)
        X1=X1.view(-1,self.f_shape1[1],self.f_shape1[2],self.f_shape1[3])
        X1=self.conv2d1(X1)
        X1 = X1.view(self.batch_nGpus, -1)
        X1=self.fcseq(X1)
        return X1


def test_and_show(args, mydata=None, colormap=None, test_all_fig=False): #最后用所有数据在训练好的网络上进行分类测试，绘制试验结果图
    if mydata is None:
        image_raw, labels_raw, edge_dic= load_data(args.data_name)
        remain_index = np.where(labels_raw != 0)  # ([h1,h2,h3,...],[w1,w2,w3,...])
        # zip返回元组组成的列表[(h1,w1),(h2,w2),(h3,w3),...] python需用list手动转换; zip(*)为解压，也需用list转换，即可恢复原样
        remain_index = np.array(list(zip(remain_index[0], remain_index[1])))#zip函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        # remain_index[:, 0]取第1维的所有第0个元素[h1,h1,...,h2,...]
        train_ind, test_ind, _, _ = my_train_test_split(remain_index, labels_raw[remain_index[:, 0], remain_index[:, 1]], #(data,labels,train_rate,random_state)
                                                        args.train_size, random_state=seed)
        val_ind, test_ind = train_test_split(test_ind, train_size=len(train_ind), random_state=seed)
        # val_ind=train_ind
        if test_all_fig:
            test_ind=np.where(labels_raw >= 0)
            test_ind=np.array(list(zip(test_ind[0], test_ind[1])))
        print('train samples:', len(train_ind))
        a, b = np.unique(labels_raw[train_ind[:, 0], train_ind[:, 1]], return_counts=True)
        print(list(zip(a, b)))
        print('test samples:', len(test_ind))
        a, b = np.unique(labels_raw[test_ind[:, 0], test_ind[:, 1]], return_counts=True)
        print(list(zip(a, b)))

        # image_pca=my_pca(image_raw,30)
        # seg_img_pca=my_pca(seg_img,30)

        # maybe pca,so we set name with xxx_pca
        image_pca = image_raw
        image_pca = preprocess(image_pca)

        mydata = MyData(image_pca, labels_raw,edge_dic, train_ind, test_ind, val_ind, args.patch_size, args.batch_size)
        # prepare Net
        bands_num = image_pca.shape[-1]
        myNet = MyNet(args.classnums, bands_num, args.patch_size, args.batch_size)
        if torch.cuda.device_count()>1:
            print('Use ',torch.cuda.device_count() ,' GPUs')
            myNet=nn.DataParallel(myNet)
        myNet.to(device)
        
        modelname = args.method_name + args.data_name + '_cla_lossbst' + '.model'
        print('load model from ' ,modelname)
        
        myNet.load_state_dict(torch.load(os.path.join(args.model_save_path, modelname)))

    print('test all...')
    myNet.eval()

    y_pred = []
    y_true = []
    indices=[]
    iters = mydata.all_iters
    start = time.time()
    for iter in tqdm(range(iters),desc='test all:'):
        X1, Yc1,ind= mydata.get_batch(iter,'all')

        X1 = np.transpose(X1, [0, 3, 1, 2])
        X1 = torch.from_numpy(X1).requires_grad_(True).to(device)
        X1 = X1.unsqueeze(1)  # 扩展维度，三维卷积输入为五维
        output = myNet(X1,X1,isTest=True)

        y_pred.extend(torch.argmax(output.detach(), 1).cpu().numpy().tolist())
        y_true.extend(Yc1.tolist())
        indices.extend(ind.tolist())

    end = time.time()
    print("infer time:", end - start)
    f = open('log.txt', 'a')
    f.write(time.asctime() + ' '*5 + args.data_name + "  inf time:  " + str(end - start) + '\n')
    f.write('-'*100 + '\n')
    f.close()


    indices=np.array(indices)
    labels_show=np.zeros(mydata.image_shape[0:2])
    labels_show[indices[:,0],indices[:,1]]=np.array(y_pred)+1
    if test_all_fig:
        allfigName = os.path.join(result_path, modelname.split('.')[0] + "allfig_single.png")
        saveFig(labels_show, allfigName)

    all_oa=metrics.accuracy_score(y_true,y_pred)
    print(all_oa)
    writelogtxt(args.logFile,'OA:'+str(all_oa))
    print_metrics(y_true,y_pred)
    figname = os.path.join(result_path, modelname.split('.')[0] + "_single.png")
    print("save ",figname)
    writelogtxt(args.logFile, 'save\r' + figname)
    saveFig(labels_show,figname)
    figname = os.path.join(result_path, modelname.split('.')[0] + "_gt.png")
    saveFig(labels_raw,figname)


def test_val(mydata:MyData, myNet:MyNet, fortype='test'):  # 使用测试集或验证集用于分类网络的测试
    print(fortype,'...')
    myNet.eval()

    y_pred=[]
    y_true=[]
    if fortype=='test':
        iters=mydata.test_iters
    elif fortype=='val':
        iters=mydata.val_iters
    else:
        print("error test_val args:fortype.")
        exit()

    for iter in range(iters):
        X1, Yc1,ind= mydata.get_batch(iter,fortype)
        X1 = np.transpose(X1, [0, 3, 1, 2])
        X1 = torch.from_numpy(X1).requires_grad_(True).to(device)
        X1 = X1.unsqueeze(1)
        output = myNet(X1,X1,isTest=True)

        y_pred.extend(torch.argmax(output.detach(),1).cpu().numpy().tolist())
        y_true.extend(Yc1.tolist())
    acc=metrics.accuracy_score(y_true,y_pred)
    return acc,y_pred,y_true


def run(args=None):

    print("run with args dict:", args.__dict__)  # 打印设置好的超参数
    # load data
    image_raw, labels_raw, edge_dic= load_data(args.data_name)  # 加载已存在的3个文件
    remain_index = np.where(labels_raw != 0)  # 返回标签值不为0的坐标，即有标签样本的坐标
    remain_index = np.array(list(zip(remain_index[0], remain_index[1])))  # ([x1,y1],[x2,y2],...)

    train_ind, test_ind, _, _ = my_train_test_split(remain_index, labels_raw[remain_index[:, 0], remain_index[:, 1]],
                                                    args.train_size, random_state=seed)
    
    val_ind, test_ind = train_test_split(test_ind, train_size=len(train_ind), random_state=seed)
    # val_ind=train_ind
    print('train samples:', len(train_ind))
    a, b = np.unique(labels_raw[train_ind[:, 0], train_ind[:, 1]], return_counts=True)
    print(list(zip(a, b)))
    print('test samples:', len(test_ind))
    a, b = np.unique(labels_raw[test_ind[:, 0], test_ind[:, 1]], return_counts=True)
    print(list(zip(a, b)))
    
    # image_pca=my_pca(image_raw,30)
    # seg_img_pca=my_pca(seg_img,30)

    # maybe pca,so we set name with xxx_pca
    image_pca = image_raw
    image_pca = preprocess(image_pca)

    mydata = MyData(image_pca, labels_raw, edge_dic, train_ind, test_ind, val_ind, args.patch_size, args.batch_size)

    # prepare Net
    bands_num = image_pca.shape[-1]
    myNet = MyNet(args.classnums, bands_num, args.patch_size, args.batch_size)
    if torch.cuda.device_count()>1:
        print('Use ',torch.cuda.device_count() ,' GPUs')
        myNet = nn.DataParallel(myNet)
    myNet.to(device)
    if args.checkpoint == True:  # checkpoint检查点：不仅保存模型的参数，优化器参数，还有loss，epoch等
        modelname = args.method_name + args.data_name + '_lossbst' + '.model'
        if os.path.exists(os.path.join(args.model_save_path, modelname)):
            myNet.load_state_dict(torch.load(os.path.join(args.model_save_path, modelname)))  # 加载训练好的参数到模型
            print('load model from ',modelname)
            writelogtxt(args.logFile,'load model from '+modelname)
    
    MSE = nn.MSELoss()
    CE = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(myNet.parameters(), lr=args.lr, momentum=0.9)

    print("Model's state_dict:")
    for param_tensor in myNet.state_dict():
        print(param_tensor, "\t", myNet.state_dict()[param_tensor].size())

    total = sum([param.nelement() for param in myNet.parameters()])
    f = open('log.txt', 'a')
    f.write("total parameters: " + str(total) + '\n')
    f.close()

    loss_bst=-1
    bst_epoch = 0

    start = time.time()
    # first stage: train the regressive net with MSE, to extract the similarity of HSI
    # here we load reg model without checkpoint-check.
    modelname = args.method_name + args.data_name + '_lossbst' + '.model'
    if os.path.exists(os.path.join(args.model_save_path, modelname)):
            myNet.load_state_dict(torch.load(os.path.join(args.model_save_path, modelname)))
            print('load reg model from ', modelname)
            writelogtxt(args.logFile, 'load reg model from: '+modelname)
    else:  # 训练过程及相关信息的打印输出与记录
        print('run reg train start:', time.asctime())
        for epoch in range(args.epochs):
            myNet.train()
            mydata.shuffle()
            for iter in range(mydata.reg_iters):  # （取几次）
                optim.zero_grad()
                X1, Yc1, X2, Yc2, Yr = mydata.get_batch_reg(iter)  # 回归训练，Yr为超像素分割后边权图的相似性标签  （每次取batch）

                X1 = np.transpose(X1, [0, 3, 1, 2])
                X2 = np.transpose(X2, [0, 3, 1, 2])
                X1 = torch.from_numpy(X1).requires_grad_(True).to(device)
                X2 = torch.from_numpy(X2).requires_grad_(True).to(device)
                X1 = X1.unsqueeze(1)
                X2 = X2.unsqueeze(1)
                outYc1, outYc2, outYr = myNet(X1, X2)
                Yr = torch.from_numpy(Yr).float().to(device)
                loss = MSE(outYr, Yr)  # 回归训练，未使用原标签数据
                loss.backward()
                optim.step()
                logtext = '\repoch_reg:{}/{} iter:{}/{} loss:{:.6f}'.format(epoch, args.epochs, iter, mydata.reg_iters, loss.detach().cpu())
                print(logtext, end='')

            logtext = '\repoch_reg:{}/{} iter:{}/{} loss:{:.6f}'.format(epoch, args.epochs, iter, mydata.reg_iters, loss.detach().cpu())
            print(logtext, end='')
            writelogtxt(args.logFile, logtext[1:])
            if (epoch + 1) % 1 == 0:
                losscpu = loss.detach().cpu()
                loss_bst = losscpu if loss_bst == -1 else loss_bst
                if losscpu <= loss_bst:
                    loss_bst = losscpu
                    bst_epoch = epoch + 1
                    modelname = args.method_name + args.data_name + '_lossbst' + '.model'
                    torch.save(myNet.state_dict(), os.path.join(args.model_save_path, modelname))
                    print("save regressive model:", modelname)
                    writelogtxt(args.logFile, "save regressive model:" + modelname)
                if epoch - bst_epoch > args.early_stop_epoch:
                    print('train regressive net break since val_acc does not improve from epoch: ', bst_epoch)
                    writelogtxt(args.logFile, 'train regressive net break since val_acc does not improve from epoch: ' + str(bst_epoch))
                    break
    end = time.time()
    f = open('log.txt', 'a')
    f.write(time.asctime() + '   ' + args.data_name + "  reg_train time:  " + str(end - start) + '\n')
    f.close()

    start_2 = time.time()
    # second stage: train the classification net with CE
    print('run cla train start:', time.asctime())
    loss_bst = -1
    bst_epoch = 0
    for epoch in range(cls_epoch):
        myNet.train()
        mydata.shuffle()
        for iter in range(mydata.all_iters):  # 在所有样本中进行训练
            optim.zero_grad()
            X1, Yc1, ind = mydata.get_batch(iter, 'all')  # 获取第iter批次的所有样本，不再是回归训练了，ind为样本数据（批次中次序，样本数据，样本标签）
            for ind_i, ind_case in enumerate(ind):
                if ind_case not in mydata.train_ind:
                    Yc1[ind_i] = -1  # 非训练样本
            X1 = np.transpose(X1, [0, 3, 1, 2])
            X1 = torch.from_numpy(X1).requires_grad_(True).to(device)
            X1 = X1.unsqueeze(1)
            output = myNet(X1, X1, isTest=True)  # isTest flag just indicates the single branch net,maybe not for test

            
            # 处理无标签分类结果,两种选择：*1认为模型分的都对，2认为模型分的都错
            tmp = np.where(Yc1 < 0)  # 非训练样本位置
            
            Yc1[tmp] = np.argmax(output.detach().cpu(), axis=1)[tmp]  # 使用模型选择标签替代非训练样本的原有标签数据
            Yc1 = torch.from_numpy(Yc1).long().to(device)
            loss = CE(output, Yc1)  # 训练样本的标签都是原有标签，非训练样本的标签都是模型选择标签，且都认为是正确的，计算损失的过程等效于将模型选择标签样本集与训练集混合后计算
            loss.backward()
            optim.step()
            logtext = '\repoch_cla:{}/{} iter:{}/{} loss:{:.6f}'.format(epoch, args.epochs, iter, mydata.all_iters, loss.detach().cpu())
            print(logtext, end='')

        logtext = '\repoch_cla:{}/{} iter:{}/{} loss:{:.6f}'.format(epoch, args.epochs, iter, mydata.all_iters, loss.detach().cpu())
        print(logtext, end='')
        writelogtxt(args.logFile, logtext[1:])

        if (epoch + 1) % 1 == 0:
            losscpu = loss.detach().cpu()
            loss_bst = losscpu if loss_bst == -1 else loss_bst
            if losscpu <= loss_bst:
                loss_bst = losscpu
                bst_epoch = epoch + 1
                modelname = args.method_name + args.data_name + '_cla_lossbst' + '.model'
                torch.save(myNet.state_dict(), os.path.join(args.model_save_path, modelname))
                print("save cla model:", modelname)
                writelogtxt(args.logFile, "save cla model:"+ modelname)
            if epoch - bst_epoch > args.early_stop_epoch:
                print('train classification net break since val_acc does not improve from epoch: ', bst_epoch)
                writelogtxt(args.logFile, 'train classification net break since val_acc does not improve from epoch: '+str(bst_epoch))
                break

    print('run cla train end:',time.asctime())

    end_2 = time.time()
    f = open('log.txt', 'a')
    f.write(time.asctime() + ' '*3 + args.data_name + "  cla_train time:  " + str(end_2 - start_2) + '\n')
    f.close()

    # test on best model
    modelname = args.method_name + args.data_name + '_cla_lossbst' + '.model'
    myNet.load_state_dict(torch.load(os.path.join(args.model_save_path, modelname)))
    test_acc, y_pred, y_true = test_val(mydata, myNet, 'test')
    print('test_acc:', test_acc)
    writelogtxt(args.logFile,'test_acc:'+str(test_acc))

    overall_accuracy, average_accuracy, kp, classify_report, confusion_matrix, acc_for_each_class = print_metrics(
        y_true, y_pred)

    a, b = np.unique(y_true, return_counts=True)
    print('y_true cnt:', list(zip(a, b)))
    a, b = np.unique(y_pred, return_counts=True)
    print('y_pred cnt:', list(zip(a, b)))
    return overall_accuracy, average_accuracy, kp, acc_for_each_class


def writelogtxt(logfilepath,content,with_time=True):
    f = open(logfilepath, 'a')
    tm_str=""
    if with_time:
        tm_str = str(time.asctime())+" "
    f.write(tm_str+content+'\n')
    f.flush()
    f.close()


class Args():
    def __init__(self):
        self.method_name = 'SLRSS_'
        self.data_name = pargs.data_name
        self.batch_size = 1024
        self.epochs = 200
        self.lr=0.0005
        self.runs = pargs.runs
        # 在set_dataset()中，相关参数会更新为命令行指定参数，所以放在最后
        self.set_dataset(self.data_name)
        self.model_save_path = os.path.join(basedir,'models')

        self.early_stop_epoch = 30
        self.checkpoint=True
        self.record = True
        #self.recordFile = self.method_name+'_'+self.data_name+'_results.txt'
        self.logfilepath=os.path.join(basedir,'logs')
        self.logFile=os.path.join(self.logfilepath,self.method_name+self.data_name+'_logs.txt')


    def set_dataset(self, data_name):
        if data_name == 'PaviaU':
            self.data_name = 'PaviaU'
            self.train_size = 5
            self.classnums = 9
            self.patch_size = 5
        elif data_name == 'IndianPines':
            self.data_name = 'IndianPines'
            self.train_size = 5
            self.classnums = 16
            self.patch_size = 5
        elif data_name == 'Salinas':
            self.data_name = 'Salinas'
            self.train_size = 5
            self.classnums = 16
            self.patch_size = 5
        elif data_name == 'Houston2013':
            self.data_name = 'Houston2013'
            self.train_size = 5
            self.classnums = 15
            self.patch_size = 5
        elif data_name == 'LongKou':
            self.data_name = 'LongKou'
            self.train_size = 5
            self.classnums = 9
            self.patch_size = 5

        # 如果参数在命令行中指定，则将参数更新为命令行指定参数
        if pargs.train_size != -1:
            self.train_size=pargs.train_size
            if self.train_size>=1:
                self.train_size=int(self.train_size)
        if pargs.patch_size != -1:
            self.patch_size=pargs.patch_size
        if pargs.epochs != -1:
            self.epochs=pargs.epochs
        if pargs.batch_size != -1:
            self.batch_size=pargs.batch_size


def run_more(args:Args):
    
    writelogtxt(args.logFile, "run more start!")
    print("run more start! ")

    global  seed
    oas = []
    aas = []
    kps = []
    afes = []
    for run_id in range(args.runs):  # 打印出并写入每一遍的oa、aa、kappa、acc_for_each_class值
        seed = np.random.randint(10000)
        # as we need to reload the reg net ,but train a new cla net,so we do not reset checkpoint
        args.checkpoint = False
        print(" reset the checkpoint to False! ")
        oa, aa, kp, acc_for_each_class = run(args)
        print('run_id:', run_id)
        print('oa:{} aa:{} kp:{}'.format(oa, aa, kp))
        writelogtxt(args.logFile, "run id:"+str(run_id))
        writelogtxt(args.logFile, 'oa:{} aa:{} kp:{}'.format(oa, aa, kp))
        oas.append(oa)
        aas.append(aa)
        kps.append(kp)
        afes.append(acc_for_each_class)
    # 打印出并写入oa、aa、kappa、acc_for_each_class值列表
    print('afes', afes)
    print('oas:', oas)
    print('aas:', aas)
    print('kps', kps)
    writelogtxt(args.logFile, "afes: "+str(afes))
    writelogtxt(args.logFile, "oas: "+str(oas))
    writelogtxt(args.logFile, "aas: "+str(aas))
    writelogtxt(args.logFile, "kps: "+str(kps))
    # 打印出并写入oa,aa,kappa值的均值和标准差
    print('mean and std:oa,aa,kp')
    print(np.mean(oas) * 100, np.mean(aas) * 100, np.mean(kps) * 100)
    print(np.std(oas), np.std(aas), np.std(kps))
    print('mean/std oa for each class,axis=0:')
    writelogtxt(args.logFile, "mean:oa,aa,kp: "+str(np.mean(oas) * 100)+" "+str(np.mean(aas) * 100)+" " + str(np.mean(kps) * 100))
    writelogtxt(args.logFile, "std:oa,aa,kp: "+str(np.std(oas) * 100)+" "+str(np.std(aas) * 100)+" " + str(np.std(kps) * 100))
    # 打印出并写入每一类的平均精度  开头不写入当前时间了
    m_afes = np.mean(afes, axis=0)
    for a in m_afes:
        print(a * 100)
        writelogtxt(args.logFile, str(a * 100),with_time=False)
    # 打印oa,aa,kappa值的均值和标准差, 以及每一类精度的方差
    print(np.mean(oas) * 100)
    print(np.mean(aas) * 100)
    print(np.mean(kps) * 100)
    print(np.std(oas) * 100)
    print(np.std(aas) * 100)
    print(np.std(kps) * 100)
    print(np.std(afes, axis=0))
    writelogtxt(args.logFile, "run more over !!")


if __name__ == '__main__':
    args=Args()  # 参数初始化
    if pargs.isTest:  # 测试时
        writelogtxt(args.logFile," a new test all \n"+'*'*50)
    else:             # 训练时
        writelogtxt(args.logFile, " a new start \n" + '*' * 50)
    writelogtxt(args.logFile,str(args.__dict__))
    if not pargs.isTest:  # 训练时
        print('run start:',time.asctime())
        run_more(args)
        print('run end:',time.asctime())

    test_and_show(args,test_all_fig=False)
