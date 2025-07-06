import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split




def load_data ( file_path ):
    return pd.read_csv(file_path)



def preprocess_data (data):
    sex = pd.get_dummies(data['person_gender'],drop_first=True,dtype=int)
    previous_loan_defaults_on_file = pd.get_dummies(data['previous_loan_defaults_on_file'],drop_first=True,dtype=int)
    data.drop(['person_gender','previous_loan_defaults_on_file'],axis=1,inplace=True)
    name_col = data.select_dtypes(include=object)


    le = LabelEncoder()

    for i in name_col:
        data[i] = le.fit_transform(data[i])


    data['person_income'] = np.log1p(data['person_income'])
    data['loan_amnt'] = np.log1p(data['loan_amnt'])

    loan_status = data['loan_status']

    data.drop('loan_status',axis=1,inplace = True)
    data =pd.concat([data,sex,previous_loan_defaults_on_file,loan_status],axis=1)
    data.rename(columns={'Yes':'previous_loan_defaults_on_file'},inplace=True)

    return data

def split_data(X , y , test_size =0.20):
    return train_test_split(X, y, test_size=test_size, random_state=42)
