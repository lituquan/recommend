#coding:utf-8
'''
	1.参考项亮《推荐系统实战》
	2.https://github.com/Lockvictor/MovieLens-RecSys
	3.数据集:ml-100k
	4.环境为：python3.5.2
'''
import random
import math 
import sys
import os 

Reccommend=dict()
UserSelect=20
ItemSelect=20
ReccommendSelect=10

#交叉训练,划分数据集合
def SplitData(data,M,k,seed):
    test=[]
    train=[]
    random.seed(seed)
    for user,item in data:
        if random.randint(0,M)==k:
            test.append([user,item])
        else:
            train.append([user,item])
    return train,test            

#计算召回率
def Recall(train,test,N):
    hit=0
    allRec=0
    for user in train.keys():
        tu=test[user]
        rank=GetRecommandation(user,N)  
        for item ,pui in rank:
            if item in tu:
                hit+=1
        allRec+=len(tu)       
    return hit/(allRec*1.0)

#计算准确率
def Precision(train,test,N):
    hit=0
    allPre=0
    for user in train.keys():
        tu=test[user]
        rank=GetRecommandation(user,N)
        for item ,pui in rank:
            if item in tu:
                hit+=1
        allPre+=len(rank)
    return hit/(allPre*1.0)

#计算覆盖率
def Coverage(train,test,N):
    recommend_items=set()
    all_items=set()
    for user in train.keys():
        for item in train[user]:
            all_items.add(item)
        rank=GetRecommandation(user,N)
        for item,pui in rank:
            recommend_items.add(item)  
    return len(recommend_items)*1.0/len(all_items)   

#计算新颖度
def Populartity(train,test,N):
    item_populartity=dict()
    for user,items in train.items():
        for item in items.keys():
            if item not in item_populartity:
                item_populartity[item]=0
            item_populartity[item]+=1
    ret=0
    n=0    
    for user in train.keys():
        rank=GetRecommandation(user,N)
        for item,pui in rank:
            ret+=math.log(1+item_populartity[item])
            n+=1
    ret/=n*1.0
    return ret                

'''
基于相似用户的推荐
1.找用户x的相似用户集合X：先计算所有用户间的相似度,两两之间计算,会很大计算量。
2.找到X->I用户喜欢的物品集合，但是用户没有听说过的物品i

1.计算相似度：余弦计算，item(a)*item(b)/(|item(a)|*|item(b)|)

2.倒排计算：
    因为很多item(a)*item(b)=0
    C[a][b]=|item(a)*item(b)|  //代表用户两个a和b交集的大小 ，是同属于a,b物品的个数
    a和b的共同兴趣=（a  b 交集）：(a b 合集)

'''
def UserSimilarity(train):
    item_users=dict()
    for u,items in train.items():
        for i in items.keys(): #遍历用户的项
            if i not in item_users:
                item_users[i]=set()
            item_users[i].add(u)#i物品增加用户u
    C=dict()
    N=dict()
    for i,users in item_users.items():
        for u in users:
            if u not in N.keys():
                N[u]=0
            if u not in C.keys():
                C[u]=dict()    
            N[u]+=1 #用户u的物品数
            for v in users:
                if u==v:
                    continue 
                if v not in C[u].keys():
                    C[u][v]=0  
                C[u][v]+=1 #用户u和v交集增加1个物品
    W=dict()
    for u,related_users in C.items():
        W[u]=dict()        
        for v,cuv in related_users.items():
            W[u][v]=cuv/math.sqrt(N[u]*N[v]) #用户u,v的相似度
    return W                

#返回一个u对i评分字典
def UserBase(train,K=UserSelect):
    #相似度计算  W[u][v]表示u和v的相似程度
    W=UserSimilarity(trainset)
    rank=dict()
    for user,interacted_items in train.items():
        rank[user]=dict()
        similarityUser=sorted(W[user].items(),key=lambda item:item[1],reverse=True)[0:K]
        for v,wuv in similarityUser: #lambda item:item[1]
            for i,rvi in train[v].items():
                if i in interacted_items:#排除u已有兴趣
                    continue
                if i not in rank[user].keys():
                    rank[user][i]=0.0                          
                rank[user][i]+=wuv*rvi #rvi 是用户v对i的评分                          
    return rank 

def ItemSimilarity(train):
    #calculate co-rated users between items
    C = dict()
    N = dict()
    for u, items in train.items():
        for i in items:
            if i not in C.keys():
                C[i]=dict()
            if i not in N.keys():
                N[i]=0
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                if j not in C[i].keys():
                    C[i][j]=0    
                C[i][j] += 1#i和j共同物品增加1个用户
    #calculate finial similarity matrix W
    W = dict()
    for i,related_items in C.items():
        W[i]=dict()
        for j, cij in related_items.items():            
            W[i][j] = cij / math.sqrt(N[i] * N[j])
    return W

def ItemBase(train , K=ItemSelect):
    W=ItemSimilarity(train)
    rank=dict()
    for user_id,ru in train.items():
        rankUser = dict()
        for i,pi in ru.items():
            for j, wj in sorted(W[i].items(), key=lambda item:item[1], reverse=True)[0:K]:
                if j in ru:
                    continue
                if j not in rankUser.keys():
                    rankUser[j] = 0.0
                rankUser[j] += pi * wj
        rank[user_id]=rankUser
    return rank

'''
    数据导入
'''
def loadfile(filename):
    ''' load a file, return a generator. '''
    fp = open(filename, 'r')
    for i, line in enumerate(fp):
        yield line.strip('\r\n')
        if i % 100000 == 0:
            print('loading %s(%s)' % (filename, i), file=sys.stderr)
    fp.close()
    print('load %s succ' % filename, file=sys.stderr)

def generate_dataset(filename, pivot=0.7):
    ''' load rating data and split it to training set and test set '''
    trainset={}
    testset={}
    trainset_len = 0
    testset_len = 0

    for line in loadfile(filename):
        user, movie, rating, _ = line.split('\t')
        # split the data by pivot
        if random.random() < pivot:
            trainset.setdefault(user, {})
            trainset[user][movie] =1 #int(rating)
            trainset_len += 1
        else:
            testset.setdefault(user, {})
            testset[user][movie] =1 #int(rating)
            testset_len += 1

    print ('split training set and test set succ', file=sys.stderr)
    print ('train set = %s' % trainset_len, file=sys.stderr)
    print ('test set = %s' % testset_len, file=sys.stderr)
    return trainset,testset

def GetRecommandation(user,N):
    return  sorted(Reccommend[user].items(),key=lambda item:item[1],reverse=True)[0:N]

    
if __name__=='__main__':
    #数据集
    ratingfile = os.path.join('ml-100k', 'u.data')
    trainset,testset=generate_dataset(ratingfile)
    #推荐    
    #Reccommend=UserBase(trainset)
    Reccommend=ItemBase(trainset)
    #参数计算
    K=ReccommendSelect
    rec=Recall(trainset,testset,K)
    pre=Precision(trainset,testset,K)
    converage=Coverage(trainset,testset,K)
    populartity=Populartity(trainset,testset,K)
    print(rec,pre,converage,populartity)