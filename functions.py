from sklearn.model_selection import train_test_split
import pandas as pd

def split_data_with_replacement(data, ratio,idx):
    data0 = data[data[idx] == 0]
    data1 = data[data[idx] == 1]
    
    

def split_data_evenly(data, ratio,idx):
    """
    this function ensures that the classes are equalized in the training and testing sets,
    where equal means that the proportion of the classes in the training and testing sets is equal.
    """
    data0 = data[data[idx] == 0]
    data1 = data[data[idx] == 1]

    # make sets equal size
    data0 = data0.sample(data1.shape[0])

    # # create a training and test set
    data_train, data_test = train_test_split(pd.concat([data0, data1]), test_size=ratio)

    X_train = data_train.iloc[:, :idx].values
    Y_train = data_train.iloc[:, idx].values
    # print(X_train.shape, Y_train.shape)

    X_test = data_test.iloc[:, :idx].values
    Y_test = data_test.iloc[:, idx].values
    # print(X_test.shape, Y_test.shape)
    return X_train, X_test, Y_train, Y_test
    
def split_data_even_proportion(data, ratio,idx):
    """
    this function ensures that the classes are balanced in the training and testing sets,
    where balance means that the proportion of the classes in the training and testing sets is the same as in the original data.
    """
    data0 = data[data[idx] == 0]
    data1 = data[data[idx] == 1]

    # instead of picking only part of the data, we can also use all of it, ensuring that the classes are balanced

    # create a training and test set for each outcome
    data0_train, data0_test = train_test_split(data0, test_size=ratio)
    data1_train, data1_test = train_test_split(data1, test_size=ratio)

    data_train = pd.concat([data0_train, data1_train]).sample(frac=1)
    data_test = pd.concat([data0_test, data1_test]).sample(frac=1) # shuffled

    X_train = data_train.iloc[:, :-1].values
    Y_train = data_train.iloc[:, -1].values
    # print(X_train.shape, Y_train.shape)

    X_test = data_test.iloc[:, :-1].values
    Y_test = data_test.iloc[:, -1].values
    # print(X_test.shape, Y_test.shape)
    
    return X_train, X_test, Y_train, Y_test

def split_data(data, ratio=0.33, idx=None):
    """
    Split the data into training and testing sets.
    :param data: a 2D numpy array
    :param ratio: the ratio of testing data
    :param idx: the index of the column that contains the target variable
    :return: X_train, Y_train, X_test, Y_test
    # """
    if idx is None:
        idx = data.shape[1] - 1
    # return split_data_evenly(data, ratio, idx)
    return split_data_even_proportion(data, ratio, idx)
    
    
def fraction_fitting(data, model, fraction, iter=5):
    """
    Split the data into training and testing sets, use only a fraction of the training data to fit the model.
    :param data: a 2D numpy array
    :param fraction: the fraction of the training data to use
    
    returns mean of iter runs over random splits of the data
    
    returns: accuracy, precision, recall of the test set
    """
    accuracy = 0
    precision = 0
    recall = 0
    for i in range(iter):
        X_train, X_test, Y_train, Y_test = split_data(data)
        
        X_train = X_train[:int(fraction*len(X_train))]
        Y_train = Y_train[:int(fraction*len(Y_train))]
        model.fit(X_train, Y_train)

        Y_hat_train = model.predict(X_train)
        Y_hat_test = model.predict(X_test)

        accuracy_test, precision_test, recall_test = model.accuracy(Y_hat_test, Y_test)

        accuracy += accuracy_test
        precision += precision_test
        recall += recall_test
    
    return accuracy/iter, precision/iter, recall/iter        

