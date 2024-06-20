import streamlit as st 
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sklearn
print(sklearn.__version__)

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('/content/heart.csv')

# print first 5 rows of the dataset
heart_data.head()

# print last 5 rows of the dataset
heart_data.head()

# number of rows and column in the dataset
heart_data.shape

# getting some info about the data
heart_data.info()

# checking for missing values
heart_data.isnull().sum()

# statistical measures about the data
heart_data.describe()

# checking the distribution of HeartDisease Variable
heart_data['HeartDisease'].value_counts()

X = heart_data.drop(columns='HeartDisease', axis=1)
Y = heart_data['HeartDisease']

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# Mengkonversi Nilai String Menjadi Tipe Data Numerik

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Contoh data X dan Y
data_X = {
    'Age': [40, 49, 37, 48, 54, 45, 68, 57, 57, 38],
    'Sex': ['M', 'F', 'M', 'F', 'M', 'M', 'M', 'M', 'F', 'M'],
    'ChestPainType': ['ATA', 'NAP', 'ATA', 'ASY', 'NAP', 'TA', 'ASY', 'ASY', 'ATA', 'NAP'],
    'RestingBP': [140, 160, 130, 138, 150, 110, 144, 130, 130, 138],
    'Cholesterol': [289, 180, 283, 214, 195, 264, 193, 131, 236, 175],
    'FastingBS': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'RestingECG': ['Normal', 'Normal', 'ST', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'LVH', 'Normal'],
    'MaxHR': [172, 156, 98, 108, 122, 132, 141, 115, 174, 173],
    'ExerciseAngina': ['N', 'N', 'N', 'Y', 'N', 'N', 'N', 'Y', 'N', 'N'],
    'Oldpeak': [0.0, 1.0, 0.0, 1.5, 0.0, 1.2, 3.4, 1.2, 0.0, 0.0],
    'ST_Slope': ['Up', 'Flat', 'Up', 'Flat', 'Up', 'Flat', 'Flat', 'Flat', 'Flat', 'Up']
}

data_Y = {
    'HeartDisease': [0, 1, 0, 1, 0, 1, 1, 1, 1, 0]
}

# Membuat DataFrame
df_X = pd.DataFrame(data_X)
df_Y = pd.DataFrame(data_Y)

# Menggunakan LabelEncoder untuk kolom Sex pada data X
label_encoder = LabelEncoder()
df_X['Sex'] = label_encoder.fit_transform(df_X['Sex'])

# Memisahkan fitur dan label dari data X dan Y
X = df_X[['Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']]
y = df_Y['HeartDisease']

# Membagi dataset menjadi data pelatihan dan pengujian
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model regresi logistik
model = LogisticRegression(max_iter=200)

# Melatih model dengan data pelatihan
model.fit(X_train, Y_train)

# Memprediksi label pada data pengujian
predictions = model.predict(X_test)

# Menampilkan prediksi
print("Prediksi:", predictions)


model = LogisticRegression()

# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

import numpy as np

# Data input dalam bentuk string
input_data_str = '37,M,ATA,130,283,0,ST,98,N,0,Up,0'

# Pisahkan string input menjadi nilai individual
input_data_list = input_data_str.split(',')

# Membuat peta (mapping) untuk konversi nilai string ke numerik
sex_mapping = {'M': 1, 'F': 0}

# Konversi nilai 'M' menjadi 1 (atau sesuai dengan mapping yang Anda tentukan)
input_data_list[1] = sex_mapping.get(input_data_list[1], input_data_list[1])

# Konversi semua nilai string yang tersisa menjadi tipe data numerik
input_data_numerical = []
for item in input_data_list:
    try:
        # Coba mengonversi item menjadi float
        input_data_numerical.append(float(item))
    except ValueError:
        # Jika gagal, tetapkan nilainya (misalnya untuk string seperti 'ATA', 'ST', 'N', 'Up')
        input_data_numerical.append(item)

# Konversi list menjadi array numpy
input_data_as_numpy_array = np.asarray(input_data_numerical)

# Cetak hasilnya
print("Data input setelah konversi:")
print(input_data_as_numpy_array)





