import pandas as pds
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.metrics import AUC


def get_dataset():
    df = pds.read_csv("Churn_Modelling.csv")
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

    df = pds.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

    x = df.drop(columns=['Exited'])
    y = df['Exited']
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=41, test_size=0.2)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train.to_numpy(), y_test.to_numpy()


def get_model():
    model = Sequential()

    model.add(Dense(10, kernel_initializer='normal', activation='relu', input_shape=(10,)))

    model.add(Dropout(rate=0.1))
    model.add(BatchNormalization())

    model.add(Dense(7, kernel_initializer='normal', activation='relu'))

    model.add(Dropout(rate=0.1))
    model.add(BatchNormalization())

    model.add(Dense(2, kernel_initializer='normal', activation='sigmoid'))

    model.compile(metrics=[AUC(name='AUC')], loss="binarycrossentropy")

    return model
