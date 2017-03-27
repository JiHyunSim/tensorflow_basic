#-*- coding:utf-8 -*-
import focus.filereader as fr
import numpy as np

class TrainData:

    data_array = [] #파일데이터를 배열로
    row_count = 0
    class_num = [0,0,0]

    def __init__(self,filename,class_num=[0,0,0]):
        data, self.row_count = fr.read_csv_file(filename)
        self.data_array = data
        self.class_num = class_num

    def __castInt(self,data):
        res=[]
        for a in data:
            x= list(map(int,a))
            res.append(x)

        return res

    def popDataRows(self,row_count=100):
        res=[]
        if len(self.data_array)>0:
            for cnt in range(row_count):
                o=self.data_array.pop(0)
                res.append(o)
                #마지막 로우를 뽑았으면 정지
                if self.row_count==0:
                    break
                self.row_count-=1

        return res

    def has_fetch_data(self):

        return self.row_count>0

    def fetch_train_data(self,fetch_size=100):

        res_x = []
        res_y = []
        if len(self.data_array)>0:
            for cnt in range(fetch_size):
                r = self.data_array.pop()

                #col1 onehot
                v1 = self.__one_hot_encode(int(r[0])-1,self.class_num[0])
                v2 = self.__one_hot_encode(int(r[1])-1,self.class_num[1])
                tmp = self.__castInt((r[2]).split(','))
                v3 = self.__multi_hot_encode(tmp,self.class_num[2])
                x = []
                x.extend(v1)
                x.extend(v2)
                x.extend(v3)

                res_x.append(x)
                y = self.__one_hot_encode(int(r[3]),2)
                res_y.append(y)

                self.row_count -= 1

                #마지막 로우를 뽑았으면 정지
                if self.row_count == 0:
                    break

        return res_x,res_y

    def fetch_all_data(self):
        res_x = []
        res_y = []

        while len(self.data_array)>0:
            r = self.data_array.pop()

            # col1 onehot
            v1 = self.__one_hot_encode(int(r[0]) - 1, self.class_num[0])
            v2 = self.__one_hot_encode(int(r[1]) - 1, self.class_num[1])
            tmp = self.__castInt((r[2]).split(','))
            v3 = self.__multi_hot_encode(tmp, self.class_num[2])
            x = []
            x.extend(v1)
            x.extend(v2)
            x.extend(v3)

            res_x.append(x)
            y = self.__one_hot_encode(int(r[3]), 2)
            res_y.append(y)

            self.row_count -= 1

        return res_x,res_y

    def __multi_hot_encode(self,data,num):

        #one_hot_list = self.__one_hot_encode(data, self.class_num)
        tmp = np.zeros([num],np.float32)
        #tmp.put(data,1)
        for o in data:
            tmp.put(o,1)

#        for o in one_hot_list:
#            tmp=np.add(tmp, o)
        return tmp

    def __one_hot_encode(self,x, n_classes):
        """
        One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
        : x: List of sample Labels
        : return: Numpy array of one-hot encoded labels
         """
        return np.eye(n_classes)[x]