import os
import cv2
import numpy as np
from colorama import Fore, init
from sklearn.svm import  SVC
import matplotlib.pyplot as plt

init()

class SupportVectorMachines:
    def __init__(self,trainpath='./datamnist_train',testpath='./datamnist_test'):
        self.trainpath=trainpath
        self.testpath=testpath

    def Dataset(self):
        Train_X,Train_Y,Test_X,Test_Y=[],[],[],[]
        Trainpath,Testpath=self.trainpath,self.testpath
        for item in os.listdir(Trainpath):
            for value in os.listdir(Trainpath+'/'+item+'/'):
                im=np.array(cv2.imread(Trainpath+'/'+item+'/'+value,-1))
                IMG=im.flatten().tolist()
                if len(IMG)!=784:
                    print("图片读入错误，错误图片的路径为")
                    print(Trainpath+'/'+item+'/'+value)
                    break
                Train_X.append(IMG)
                Train_Y.append(int(item))

        for item in os.listdir(Testpath):
            for value in os.listdir(Testpath+'/'+item+'/'):
                im=np.array(cv2.imread(Testpath+'/'+item+'/'+value,-1))
                IMG = im.flatten().tolist()
                if len(IMG) != 784:
                    print("图片读入错误，错误图片的路径为")
                    print(Trainpath + '/' + item + '/' + value)
                    break
                Test_X.append(IMG)
                Test_Y.append(int(item))
        if len(Train_X)==len(Train_Y) and len(Test_X)==len(Test_Y):
            print(Fore.YELLOW,'文件加载完成，训练样本{}个，测试样本{}个'.format(len(Train_X),len(Test_X)))
        else:
            print(Fore.RED,"文件加载错误")
        return Train_X,Train_Y,Test_X,Test_Y

    def Trainer(self):
        X,Y,Test_X,Test_Y=SupportVectorMachines.Dataset(self)
        model=SVC(kernel='rbf')
        model.fit(X,Y)
        print(Fore.GREEN,'训练完成，正在评估得分：')
        scor=model.score(Test_X,Test_Y)
        print(Fore.BLUE,"得分评定成功，该模型得分为：{}".format(scor))
        Prediect_Y=model.predict(Test_X)
        print(Fore.CYAN,"正在画图，请等待...")
        #################画图##########################################
        plot_x=np.arange(0,1000,1000/len(Prediect_Y))
        plt.scatter(plot_x,Prediect_Y+0.02,marker='*',label='prediect')
        plt.scatter(plot_x,Test_Y,marker='.',label='source')
        plt.xlabel('number')
        plt.ylabel('value')
        plt.legend()
        plt.title("SVM")
        plt.show()
        print(Fore.LIGHTRED_EX,"本次训练结束")
        ################################################################
        return model

    def User(self,IMGpath='./usertotest/'):
        if IMGpath==None:
            print(Fore.RED,"请输入图片!!!")
        else:
            model=SupportVectorMachines.Trainer(self)
            for value in os.listdir(IMGpath):
                im = np.array(cv2.imread(IMGpath+value, -1))
                IMG = im.flatten().tolist()
                if len(IMG) != 784:
                    print(Fore.YELLOW,"图片读入错误，错误图片的路径为")
                    print(IMGpath)
                else:
                    result=model.predict([IMG])
                    print(Fore.GREEN,value+'\t'+"图片上的数字为：{}".format(result[0]))
                    image=cv2.imread(IMGpath+value)
                    cv2.imshow('',image)
                    cv2.waitKey(0)



if __name__ == '__main__':
    A=SupportVectorMachines()
    A.User()