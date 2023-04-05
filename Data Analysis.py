import numpy as np
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
import seaborn as sns

features = list()
with open('UCI_HAR_Dataset/features.txt') as f:
    features = [line.split()[1] for line in f.readlines()]
print('No of Features: {}'.format(len(features)))


def train_data():
    X_train = pd.read_csv('UCI_HAR_Dataset/train/X_train.txt', delim_whitespace=True, header=None)
    X_train.columns = [features]
    X_train['subject'] = pd.read_csv('UCI_HAR_Dataset/train/subject_train.txt', header=None, squeeze=True)
    y_train = pd.read_csv('UCI_HAR_Dataset/train/y_train.txt', names=['Activity'], squeeze=True)
    y_train_labels = y_train.map({1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', \
                                  4: 'SITTING', 5: 'STANDING', 6: 'LAYING'})

    train = X_train
    train['Activity'] = y_train
    train['ActivityName'] = y_train_labels
    train.sample(2)
    return train


train = train_data()


def test_data():
    X_test = pd.read_csv('UCI_HAR_dataset/test/X_test.txt', delim_whitespace=True, header=None)
    X_test.columns = [features]
    X_test['subject'] = pd.read_csv('UCI_HAR_dataset/test/subject_test.txt', header=None, squeeze=True)
    y_test = pd.read_csv('UCI_HAR_dataset/test/y_test.txt', names=['Activity'], squeeze=True)
    y_test_labels = y_test.map({1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', \
                                4: 'SITTING', 5: 'STANDING', 6: 'LAYING'})
    test = X_test
    test['Activity'] = y_test
    test['ActivityName'] = y_test_labels
    test.sample(2)
    return test


test = test_data()


def clean_data(train, test):
    print('No of duplicates in train: {}'.format(sum(train.duplicated())))
    print('No of duplicates in test : {}'.format(sum(test.duplicated())))
    print('We have {} NaN/Null values in train'.format(train.isnull().values.sum()))
    print('We have {} NaN/Null values in test'.format(test.isnull().values.sum()))
    train.to_csv('UCI_HAR_Dataset/csv_files/train.csv', index=False)
    test.to_csv('UCI_HAR_Dataset/csv_files/test.csv', index=False)
    train = pd.read_csv('UCI_HAR_dataset/csv_files/train.csv')
    test = pd.read_csv('UCI_HAR_dataset/csv_files/test.csv')
    print(train.shape, test.shape)
    columns = train.columns
    columns = columns.str.replace('[()]', '')
    columns = columns.str.replace('[-]', '')
    columns = columns.str.replace('[,]', '')
    train.columns = columns
    test.columns = columns
    test.columns
    return test, train


train, test = clean_data(train, test)

sns.set_style('whitegrid')
plt.figure(figsize=(16, 8))
plt.title('Data provided by each user', fontsize=20)
sns.countplot(x='subject', hue='ActivityName', data=train)
plt.show()