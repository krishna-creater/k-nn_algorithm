import numpy as np
import math
import csv

def import_data():
    train_x = np.genfromtxt('train_X_knn.csv',delimiter=',',dtype=np.float64, skip_header=1)
    # print(a.shape)
    train_Y = np.genfromtxt('train_Y_knn.csv',delimiter=',',dtype=np.float64)
    # print(b.shape)
    return train_x,train_Y

def compute_ln_norm_distance(vector1, vector2, n):
    #TODO Complete the function implementation. Read the Question text for details
    diff = [abs(vector1[i]-vector2[i]) for i in range(len(vector1))]
    ans = 0
    for item in diff:
        ans+= item**n
    # print(ans)
    k = np.power(ans, 1.0/n)
    return round(k, 4)

def find_k_nearest_neighbors(train_X, test_example, k, n):
    #TODO Complete the function implementation. Read the Question text for details
    distances = []
    for i in range(len(train_X)):
        dis = compute_ln_norm_distance(test_example, train_X[i],n)
        distances.append([dis, i])
    distances = sorted(distances, key = lambda x:x[0])
    ans = []
    for i in range(k):
        ans.append(distances[i][1])
    return ans

def classify_points_using_knn(train_X, train_Y, test_X, k, n):
    #TODO Complete the function implementation. Read the Question text for details
    ans = []
    for i in range(len(test_X)):
        k_nn = find_k_nearest_neighbors(train_X, test_X[i],k,n)
        dic = dict([])
        for i in range(len(k_nn)):
            if(train_Y[k_nn[i]] in dic):
                dic[train_Y[k_nn[i]]]+=1;
            else:
                dic[train_Y[k_nn[i]]]=1
        dic = dic.items()
        dic = sorted(dic, key = lambda x: (-x[1],x[0]))
        ans.append(dic[0][0])
    return ans

def calculate_accuracy(predicted_Y, actual_Y):
    #TODO Complete the function implementation. Read the Question text for details
    corr = 0
    for i in range(len(predicted_Y)):
        if(predicted_Y[i]==actual_Y[i]):
            corr+=1
    return corr/len(predicted_Y)



def get_best_k_using_validation_set(train_X, train_Y, validation_split_percent,n):
    #TODO Complete the function implementation. Read the Question text for details
    train_list_no = math.floor((100-validation_split_percent)*len(train_X)/100)
    validate_X = train_X[train_list_no:]
    validate_Y = train_Y[train_list_no:]
    train_X = train_X[:train_list_no]
    train_Y = train_Y[0:train_list_no]
    # print(train_list, validate_list, sep="\n")
    best_k = -1
    best_accuracy = 0
    for k in range(1, len(train_X)+1):
        predicted_Y = classify_points_using_knn(train_X, train_Y,validate_X, n, k)
        accuracy = calculate_accuracy(predicted_Y, validate_Y)
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy
    return best_k

if __name__ == "__main__":
    train_X, train_Y = import_data()
    print(get_best_k_using_validation_set(train_X,train_Y,30,3))