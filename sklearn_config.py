from tensorflow.keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten,Input,MaxPooling2D,Activation,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow import keras


#RF configs
def create_RF_MNIST() :
    model = RandomForestClassifier(max_depth=45, max_features='sqrt', min_samples_split=5,
                        n_estimators=155 , random_state=0)
    return model


def create_RF_Fashion() :
    model = RandomForestClassifier(max_depth=23, max_features='sqrt', min_samples_split=10,
                        n_estimators=200, random_state=0)
    return model


def create_RF_GTSRB() :
    model = RandomForestClassifier(max_depth=45, max_features='sqrt', min_samples_split=5,
                        n_estimators=155, random_state=0)
    return model


def create_RF_GTSRB_RGB() :
    model = RandomForestClassifier(max_depth=45, min_samples_split=5, n_estimators=200,
                        random_state=0)
    return model

def create_RF_CIFAR_RGB() :
    model = RandomForestClassifier(max_depth=45, min_samples_split=5, n_estimators=200,
                        random_state=0)
    return model

def create_RF_SignLanguage() :
    model = RandomForestClassifier(max_depth=45, min_samples_split=5, n_estimators=200,
                        random_state=0)
    return model


def create_RF_KMNIST() :
    model = GradientBoostingClassifier(learning_rate=0.05, max_depth=8, max_features='auto',
                            n_estimators=12, random_state=0)
    return model


#GB configs

def create_GB_MNIST() :
    model = GradientBoostingClassifier(learning_rate=0.15, max_depth=12,
                            max_features='sqrt', n_estimators=70 , random_state=0)
    return model


def create_GB_Fashion() :
    model = GradientBoostingClassifier(learning_rate=0.2, max_depth=7, max_features='sqrt',
                            n_estimators=50, random_state=0)
    return model


def create_GB_GTSRB() :
    model = GradientBoostingClassifier(learning_rate=0.05, max_depth=8, max_features='auto',
                            n_estimators=12, random_state=0)
    return model



def create_GB_GTSRB_RGB() :
    model = GradientBoostingClassifier(learning_rate=0.2, max_depth=8, max_features='auto',
                            n_estimators=20, random_state=0)
    return model

def create_GB_CIFAR_RGB() :
    model = GradientBoostingClassifier(learning_rate=0.2, max_depth=8, max_features='auto',
                            n_estimators=20, random_state=0)
    return model

def create_GB_SignLanguage() :
    model = GradientBoostingClassifier(learning_rate=0.2, max_depth=9, max_features='auto',
                            n_estimators=40, random_state=0)
    return model


def create_GB_KMNIST() :
    model = GradientBoostingClassifier(learning_rate=0.15, max_depth=10,
                            max_features='sqrt', n_estimators=40,
                            random_state=0)
    return model





#CNN configs

def create_CNN_MNIST() :
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_CNN_Fashion() :
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_CNN_GTSRB() :
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=(30,30,1)))
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=-1))

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))
    lr = 0.001
    epochs = 30
    opt = Adam(learning_rate=lr, decay=lr / (epochs * 0.5))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def create_CNN_GTSRB_RGB() :
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(30,30,3)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
    
    
def create_CNN_CIFAR_RGB() :
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    # compile model
    # opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    
def create_CNN_SignLanguage() :
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.20))

    model.add(Dense(24, activation = 'softmax'))

    model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    return model


def create_CNN_KMNIST() :
    #Model
    model = Sequential()
    # Add convolution 2D
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', padding="same",
            kernel_initializer='he_normal',input_shape=(28, 28, 1)))

    model.add(BatchNormalization())

    model.add(Conv2D(32,kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    # Add dropouts to the model
    model.add(Dropout(0.4))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=2,padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), strides=2,padding='same', activation='relu'))
    # Add dropouts to the model
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # Add dropouts to the model
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    # Compile the model
    model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


class MyKerasClassifier(KerasClassifier):
    def predict_proba(self, X):
        return self.model.predict(X)