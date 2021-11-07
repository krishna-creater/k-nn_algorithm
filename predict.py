import numpy as np
import csv
import sys
from train import find_k_nearest_neighbors
from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path'.
Writes the predicted values to the file named "predicted_test_Y_knn.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    return test_X


def predict_target_values(test_X):
    # print(test_X)
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    train_X = np.genfromtxt('train_X_knn.csv',delimiter=',',dtype=np.float64, skip_header=1)
    # print(train_X)
    train_Y = np.genfromtxt('train_Y_knn.csv',delimiter=',',dtype=np.float64)
    # print(b.shape)
    ans = []
    for i in range(len(test_X)):
        k_nn = find_k_nearest_neighbors(train_X, test_X[i],2,2)
        # print(k_nn)
        dic = dict([])
        for i in range(len(k_nn)):
            if(train_Y[k_nn[i]] in dic):
                dic[train_Y[k_nn[i]]]+=1;
            else:
                dic[train_Y[k_nn[i]]]=1
        dic = dic.items()
        dic = sorted(dic, key = lambda x: (-x[1],x[0]))
        print(dic)
        ans.append(int(dic[0][0]))
        # np.append(ans,dic[0][0])
    print(np.array(ans))
    return np.array(ans)

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X = import_data(test_X_file_path)
    pred_Y = predict_target_values(test_X)
    print(pred_Y)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]#'train_X_knn.csv'
    predict(test_X_file_path)
    # Uncomment to test on the training data
    #validate(test_X_file_path = 'train_X_knn.csv', actual_test_Y_file_path="train_Y_knn.csv")