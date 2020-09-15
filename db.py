# 

import pymongo
import json
import time


import matplotlib.pyplot as plt

class DB:
    def __init__(self,password="111",db_name="run_log"):
        client = pymongo.MongoClient("mongodb://terry:"+password+"@cluster0-shard-00-00.b434c.mongodb.net:27017,cluster0-shard-00-01.b434c.mongodb.net:27017,cluster0-shard-00-02.b434c.mongodb.net:27017/"+db_name+"?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin&retryWrites=true&w=majority")
        self.db = client.run_log
        # print(db)
    def add_one(self,data,name="log"):
        """
        添加一条数据
        """
        # print(type(data))
        data["_id"]=time.time()
        self.db[name].insert_one(data)
    def clear_col(self,name="log"):
        """
        清空单个表
        """
        self.db[name].remove({}) 
    def get_col(self,name="log"):
        """
        遍历表
        """
        for it in self.db[name].find({}):
            # print((it))
            yield it
if __name__ == '__main__':
    password=input("password:")
    Db=DB(password=password)
    # Db.add_one({'xx':11})
    while True:
        x=[]
        y=[]
        for i,it in enumerate( Db.get_col()):
            print(it)
            # x.append(i)
            x.append(it['step'])
            y.append(it['loss'])

        plt.figure()
        plt.plot(x,y)
        plt.xlabel("step") #X轴标签
        plt.ylabel("loss") #Y轴标签
        plt.title("run loss") #标题 
        plt.show()  #显示图
        plt.savefig("data/plot.jpg")
        time.sleep(10)
