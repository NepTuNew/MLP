import numpy as np
import pandas as pd

PATH = 'hw2_data/IRIS.TXT'
def loadDataset(PATH):
    """

    :param PATH:
    :return:
    train: the train data==> is a list store the nparray for each data, and the shape(1, input_dimension)
    one_hot: is a list store nparray for the ground truth that encode to one_hot format, and the shape(1, output)
            for example : if output=5 and data belong to class 5then [0, 0, 0, 0, 1(for class5)]
    max_value: is the max value in the dataset, and I use this value to normalize the data
    """
    data = pd.read_csv(PATH, sep=' ', header=None)
    labels = data[data.columns.shape[0]-1].unique()
    dimension = data.columns.shape[0]
    print(dimension)
    train = []
    test = []
    one_hot_train = []
    one_hot_test = []
    map_dict = {}
    # find max value in dataframe
    max_value = 0
    for i in range(len(data.columns)-1): # -1 for label
        if data[i].max() > max_value:
            max_value = data[i].max()

    for index, label in enumerate(labels):
        map_dict[label] = index
    data[dimension-1] = data[dimension-1].map(map_dict)
    if len(data) < 20:
        for i in range(data.shape[0]):
            train.append(np.array(data.iloc[i][:-1]) / max_value)
            label = np.zeros([1, labels.shape[0]])
            label[0][int(data.iloc[i][dimension - 1])] = 1
            one_hot_train.append(label)
        return train, one_hot_train, test, one_hot_test, max_value
    for i in range(data.shape[0]):
        if np.random.rand() > 0.75:
            test.append(np.array(data.iloc[i][:-1])/max_value)
            label = np.zeros([1, labels.shape[0]])
            label[0][int(data.iloc[i][dimension - 1])] = 1
            one_hot_test.append(label)
            continue
        train.append(np.array(data.iloc[i][:-1])/max_value)
        label = np.zeros([1, labels.shape[0]])
        label[0][int(data.iloc[i][dimension-1])] = 1
        one_hot_train.append(label)

    return train, one_hot_train, test, one_hot_test, max_value
    #return data, labels
if __name__ == '__main__':
    data , labels, max= loadDataset(PATH)