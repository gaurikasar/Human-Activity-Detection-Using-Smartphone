
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout
from keras.layers import BatchNormalization
from keras.regularizers import L1L2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix

activity_list = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING',
}


def new_confusion_matrix(Y_true, Y_pred):
    Y_true = pd.Series([activity_list[y] for y in np.argmax(Y_true, axis=1)])
    Y_pred = pd.Series([activity_list[y] for y in np.argmax(Y_pred, axis=1)])

    return pd.crosstab(Y_true, Y_pred, rownames=['True'], colnames=['Pred'])

    plt.show(Y_true,Y_pred)


DATADIR = 'UCI_HAR_Dataset'

# Raw data signals
# Signals are from Accelerometer and Gyroscope
# The signals are in x,y,z directions
# Sensor signals are filtered to have only body acceleration
# excluding the acceleration due to gravity
# Triaxial acceleration from the accelerometer is total acceleration
signals_list = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z"
    ]


def _read_csv(filename):
    return pd.read_csv(filename, delim_whitespace=True, header=None)

def load_signals(subset):
    signals_data = []

    for signal in signals_list:
        filename = f'C:/Users/gauri/OneDrive/Desktop/pythonProject/UCI HAR Dataset/{subset}/Inertial signals_list/{signal}_{subset}.txt'
        signals_data.append(
            _read_csv(filename).to_numpy()
        )

    # Transpose is used to change the dimensionality of the output,
    # aggregating the signals_list by combination of sample/timestep.
    # Resultant shape is (7352 train/2947 test samples, 128 timesteps, 9 signals_list)
    return np.transpose(signals_data, (1, 2, 0))

def load_y(subset):

    filename = f'C:/Users/gauri/OneDrive/Desktop/pythonProject/UCI HAR Dataset/{subset}/y_{subset}.txt'
    y = _read_csv(filename)[0]

    return pd.get_dummies(y).to_numpy()

def load_data():
    """
    Obtain the dataset from multiple files.
    Returns: X_train, X_test, y_train, y_test
    """
    X_train, X_test = load_signals('train'), load_signals('test')
    y_train, y_test = load_y('train'), load_y('test')

    return X_train, X_test, y_train, y_test

np.random.seed(42)

tf.random.set_seed(42)

# Initializing parameters
epochs = 30
batch_size = 16
n_hidden = 32

#function to count the number of classes
def _count_classes(y):
    return len(set([tuple(category) for category in y]))

X_train, X_test, Y_train, Y_test = load_data()

timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
n_classes = _count_classes(Y_train)

# sequential model initialised
model = Sequential()
# Cconfiguring parameters for model
model.add(LSTM(n_hidden, input_shape=(timesteps, input_dim)))
# Dropout layer added
model.add(Dropout(0.5))
# Dense output layer with sigmoid function.
model.add(Dense(n_classes, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_test, Y_test),epochs=epochs)
newcm = new_confusion_matrix(Y_test, model.predict(X_test))
score = model.evaluate(X_test, Y_test)
print("\n   cat_crossentropy  ||   accuracy ")
print("  ____________________________________")
print(score)
Y_pred = model.predict(X_test)
Y_true = np.argmax(Y_test, axis=1)
Y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(Y_true, Y_pred)
print('\n ********Confusion Matrix********')
print('\n {}'.format(cm))

print('****************| Classifiction Report |****************')
classification_report = metrics.classification_report(Y_true, Y_pred)
results = dict()
results['confusion_matrix'] = cm

results['classification_report'] = classification_report
print(classification_report)
