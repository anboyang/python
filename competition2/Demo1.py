from csv import reader
import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# 对数据进行标准化预处理（去均值除以标准差）
def init_data(data):
    average = []
    std = []
    for i in range(13):
        average.append(np.mean(data[:,i]))
        std.append(np.mean(data[:,i]))
    for p in range(len(data)):
        for q in range(13):
            data[p][q] = (data[p][q] - average[q]) / std[q]
    return data

# 将字符串的数据类型转换为浮点类型
def str_to_float(line):
    new_line = []
    for row in line:
        new_line.append(float(row))
    return new_line

# 读取数据，并对数据中的'?'进行处理
def load_data(filename):
    with open(filename, 'r') as file:
        data = list()
        data_reader = reader(file)
        for row in data_reader:
            line = []
            if not row:
                continue
            for chw in row:
                if chw == '?':
                    line.append('9')
                    continue
                line.append(chw)
            data.append(str_to_float(line))
    # data = init_data(np.array(data))
    data = np.array(data)
    return data

# 利用卡方检验方法进行最有特征的提取
def selectKBest(data):
    return SelectKBest(chi2, k=6).fit(data[:,:-1],data[:,-1])

# 将训练集中的数据分为训练集和测试集（用于前期自己验证正确率，调试算法模型和参数）
def dive_data(model,data):
    datas = model.transform(data[:,:-1])
    labels = data[:,-1]
    test_number = len(data) // 7
    return datas[test_number:], labels[test_number:], datas[:test_number], labels[:test_number]

# 将预测出来的类别写入csv文件中
def write_data(filename_test,forecast):
    with open(filename_test, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in forecast:
            writer.writerow(str(row))


filename = './train.csv'
datas = load_data(filename)
# 利用卡方检验的方法选出最优的6个特征
model = selectKBest(datas)
trainData, trainLabel, testData, testLabel = dive_data(model,datas)
# 运用KNN进行模型的学习与训练
clf = KNeighborsClassifier(n_neighbors=40)
clf.fit(trainData,trainLabel)
# 输出自己调试时，算出来的争取率（用于自己分析，调整算法模型和参数）
print(clf.score(testData,testLabel))


# 最终测试集的测试
train = load_data(filename)
train_data = train[:,:-1]
train_data = model.transform(train_data)
train_label = train[:,-1]
clf = KNeighborsClassifier(n_neighbors=40)
clf.fit(train_data,train_label)
filename_test = './test.csv'
data = load_data(filename_test)
data = model.transform(data)

forecast_test = clf.predict(data)

# 将预测出来的类别写入csv文件中
np.savetxt('predict.csv',forecast_test)




