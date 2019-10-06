import numpy as np
import csv

def init_data(data):
    average1 = np.mean(np.array(data)[:,0])
    average2 = np.mean(np.array(data)[:,1])
    std1 = np.std(np.array(data)[:,0])
    std2 = np.std(np.array(data)[:,1])
    for i in range(len(data)):
        data[i][0] = (data[i][0] - average1) / std1
        data[i][1] = (data[i][1] - average2) / std2
    return data

def sigmod(z):
    return (1 / (1 + np.exp(-z)))

def grad_descent(data, target):
    data = np.insert(data, 0, 1, axis=1)
    dataMat = np.mat(data)  # 将数据集转换为矩阵类型
    targetMat = np.mat(target).transpose()  # 将类别集转换为矩阵类型，再求转置矩阵
    m, n = np.shape(dataMat)     # 求出数据矩阵的维数
    weights = np.ones((n, 1))    # 初始化参数矩阵 (n 行 1 列，数据为 1 的矩阵)
    alpha = 0.001  # 学习率
    maxCycle = 500  # 学习次数

    for i in range(maxCycle):
        h = sigmod(dataMat * weights)   # 逻辑回归，通过sigmoid函数算出概率值
        # 通过梯度下降的方法(数据值减去该数值在代价函数的一阶导函数的数值)算出使得代价函数最小的预测函数的参数向量
        weights = weights - alpha * dataMat.transpose() * (h - targetMat)

    return weights

def judge(data, weights, alpha):  # alpha是判断阈值
    data = np.insert(data, 0, 1, axis=0)
    dataMat = np.mat(data)  # x矩阵
    # m, n = np.shape(dataMat)
    h = sigmod(dataMat * weights)
    if h > alpha:
        return 1
    else:
        return 0

def write_data(filename_test,forecast):
    with open(filename_test, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in forecast:
            writer.writerow(str(row))


data = np.loadtxt('./HTRU_2_train.csv', delimiter=',')
labels = data[:, -1]  # 标签集
data = data[:, :-1]  # 数据集
data = init_data(data)
test_number = len(data) // 7

testData = data[:test_number]
testLab = labels[:test_number]

traningData = data[test_number:]
traningLab = labels[test_number:]

weights = grad_descent(traningData, traningLab)
print(weights)
forecast = []
for i in range(len(testData)):
    forecast.append(judge(testData[i], weights, 0.5))

correct = 0
for index in range(len(testLab)):
    if(testLab[index] == forecast[index]):
        correct = correct + 1
    continue
print("正确率为："+ str(correct / len(testData) * 100) + "%")

filename_test = 'HTRU_2_test.csv'
data_test = np.loadtxt('./HTRU_2_test.csv', delimiter=',')
data_test = init_data(data_test)
forecast_test = []
for i in range(len(data_test)):
    forecast_test.append(judge(data_test[i], weights, 0.5))
write_data(filename_test,forecast_test)
