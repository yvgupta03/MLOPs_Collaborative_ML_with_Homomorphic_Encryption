# imports
import os
import time

import numpy as np
import pandas as pd

import boto3
from botocore.exceptions import ClientError

from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier

import pickle
import tenseal as ts
import urllib3

from collections import Counter

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding


# function to fetch private and public keys for asymmetric encryption
def get_encryption_keys():
    # read keys from remote server
    http = urllib3.PoolManager()

    bytes_private_key = http.request(
        'GET', 'https://personal.utdallas.edu/~pxn210006/keys/private_key.pem')
    bytes_public_key = http.request(
        'GET', 'https://personal.utdallas.edu/~pxn210006/keys/public_key.pem')

    private_key = serialization.load_pem_private_key(
        bytes_private_key.data,
        password=None,
        backend=default_backend()
    )

    public_key = serialization.load_pem_public_key(
        bytes_public_key.data,
        backend=default_backend()
    )

    return private_key, public_key


# function that decrypts the received model file
def decrypt_master_model(active_file, node):
    # Decrypting the model
    decrypt_file = 'decrypted_worker_model_'+str(node)
    input = open(decrypt_file, 'ab')

    # get keys
    private_key, public_key = get_encryption_keys()

    # perform decryption
    with open(active_file, 'rb') as output:
        while True:
            encrypt = output.read(256)

            if not encrypt:
                break

            original_message = private_key.decrypt(
                encrypt,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            input.write(original_message)

    input.close()


# normalization of dataframe data
def normalize_df(df):
    for column in df.columns:
        df[column] = (df[column] - df[column].min()) / \
            (df[column].max() - df[column].min())
    return df


# handling outlier data in dataframe
def outlier_detection(df, n, columns):
    rows = []
    will_drop_train = []
    for col in columns:
        Q1 = np.nanpercentile(df[col], 25)
        Q3 = np.nanpercentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_point = 1.5 * IQR
        rows.extend(df[(df[col] < Q1 - outlier_point) |
                    (df[col] > Q3 + outlier_point)].index)
    for r, c in Counter(rows).items():
        if c >= n:
            will_drop_train.append(r)
    return will_drop_train


# function that performs preprocessing of the dataset
def preprocess_data(csv_file):
    # passing address of csv file to create data frame
    df = pd.read_csv(csv_file)

    # renaming columns
    df.rename(columns={'height(cm)': 'height', 'weight(kg)': 'weight', 'waist(cm)': 'waist',
                       'eyesight(left)': 'eyesight_left', 'eyesight(right)': 'eyesight_right',
                       'hearing(left)': 'hearing_left', 'hearing(right)': 'hearing_right',
                       'fasting blood sugar': 'fasting_blood_sugar',  'Cholesterol': 'cholesterol',
                       'HDL': 'hdl', 'LDL': 'ldl', 'Urine protein': 'urine_protein',
                       'serum creatinine': 'serum_creatinine', 'AST': 'ast', 'ALT': 'alt',
                       'Gtp': 'gtp', 'dental caries': 'dental_caries'}, inplace=True)

    # converting non-numeric columns to numeric data type
    df['gender'] = df['gender'].str.replace('F', '0')
    df['gender'] = df['gender'].str.replace('M', '1')
    df['gender'] = pd.to_numeric(df['gender'])

    df['tartar'] = df['tartar'].str.replace('N', '0')
    df['tartar'] = df['tartar'].str.replace('Y', '1')
    df['tartar'] = pd.to_numeric(df['tartar'])

    df['oral'] = df['oral'].str.replace('N', '0')
    df['oral'] = df['oral'].str.replace('Y', '1')
    df['oral'] = pd.to_numeric(df['oral'])

    # cleaning data by observation
    df = df.drop(['ID'], axis=1)

    # removing oral column due to skewed data
    df = df.drop("oral", axis='columns')

    # handling outliers in df
    will_drop_train = outlier_detection(
        df, 3, df.select_dtypes(["float", "int"]).columns)
    df.drop(will_drop_train, inplace=True, axis=0)

    # creating x and y split where y is the resultant classification data
    x = df[['age', 'gender', 'height', 'weight', 'waist', 'hdl', 'ldl', 'serum_creatinine',
            'alt', 'gtp', 'dental_caries', 'tartar', 'triglyceride', 'hemoglobin']].copy()
    y = df['smoking'].copy()

    # normalizing x data to maintain the scale necessary for creation of model
    x = normalize_df(x)

    return x, y


# function that trains the model on local data
def train_model(sgd_model, x_train, y_train, print_flag=False):
    sgd_model.partial_fit(x_train, y_train, classes=np.unique(y_train))

    if print_flag:
        x_train_prediction = sgd_model.predict(x_train)
        training_data_accuracy = accuracy_score(x_train_prediction, y_train)
        print('Training data accuracy: ', training_data_accuracy)

        x_test, y_test = preprocess_data(
            'https://personal.utdallas.edu/~pxn210006/dataset/dataset_test.csv')
        score = sgd_model.score(x_test, y_test)
        print('New model accuracy on test data', score)

    return sgd_model


# functions that checks model accuracy on test data
def print_test_accuracy(sgd_model, input_csv_data):
    x_in, y_in = preprocess_data(input_csv_data)
    result = sgd_model.predict(x_in)
    print('Result for input data ', result)

    score = accuracy_score(result, y_in)
    print('Model accuracy on input data ', score)


# function that performs homomorphic encryption of model parameters
def encrypt_model_parameters(sgd_model, final_encrypted_model, node):
    # combine model params into numpy array
    sgd_params = np.hstack((sgd_model.intercept_[:, None], sgd_model.coef_))

    # define context for tenseal
    def context():
        context = ts.context(ts.SCHEME_TYPE.CKKS, 8192,
                             coeff_mod_bit_sizes=[60, 40, 40, 60])
        context.global_scale = pow(2, 40)
        context.generate_galois_keys()
        return context

    context = context()

    # encrypt parameters and convert into bytes
    sgd_params_encrypted = ts.ckks_tensor(context, sgd_params)
    params_encrypted = sgd_params_encrypted.serialize()

    tenseal_encrypt = 'tenseal_encrypted_model_'+str(node)
    with open(tenseal_encrypt, 'wb') as file:
        file.write(params_encrypted)


# function that return client object to interact with s3 bucket
def get_s3_client():
    http = urllib3.PoolManager()
    aws_access_key_id = http.request(
        'GET', 'https://personal.utdallas.edu/~pxn210006/keys/aws_access_key_id')
    aws_secret_access_key = http.request(
        'GET', 'https://personal.utdallas.edu/~pxn210006/keys/aws_secret_access_key')

    awsAccessKeyID = aws_access_key_id.data.decode()
    awsSecretAccessKey = aws_secret_access_key.data.decode()

    s3_client = boto3.client(
        "s3", aws_access_key_id=awsAccessKeyID, aws_secret_access_key=awsSecretAccessKey)

    return s3_client


# function that checks if file object present in s3 bucket
def s3_key_exists(filepath):
    s3_client = get_s3_client()
    bucketName = 'team-20-sptopic-master'

    try:
        s3_client.head_object(Bucket=bucketName, Key=filepath)
    except ClientError as e:
        return False

    return True


# function that fetches files from s3 bucket
def get_files_s3(file, local, node):
    s3_client = get_s3_client()
    bucketName = 'team-20-sptopic-master'

    if os.path.isfile(local):
        os.remove(local)

    s3_client.download_file(bucketName, file, local)


# function that sends files to s3 bucket
def store_files_s3(local_file_name, dest_file_name):
    s3_client = get_s3_client()
    bucketName = 'team-20-sptopic-master'

    s3_client.upload_file(local_file_name, bucketName, dest_file_name)


# function that deletes files from s3 bucket
def delete_file_s3(dest_file):
    s3_client = get_s3_client()
    bucketName = 'team-20-sptopic-master'

    s3_client.delete_object(Bucket=bucketName, Key=dest_file)


# function to get user consent on using data for training
def get_consent():
    x = input('Do you consent to use this data for improving the model? (Y/N) ')
    x = x.lower()
    if x == "y":
        return True
    elif x == "n":
        return False
    else:
        print("Invalid option selected. Please try again and choose 'y' or 'n' ")
        return get_consent()


# main function
def main():

    node_num = input("Node Number: ")

    active_file = 'active_worker_model_'+str(node_num)
    passive_file = 'passive_master_model'
    # active file path
    active_path = 'worker-node'+str(node_num)+'/'+active_file
    # passive path
    passive_path = 'worker-node'+str(node_num)+'/'+passive_file

    while (True):

        if s3_key_exists(active_path):
            # download file from s3
            get_files_s3(active_path, active_file, node_num)

            # after receiving model from master
            # decrypt the model
            decrypt_master_model(active_file, node_num)
            print('Model from master downloaded and decrypted')

            # delete the model
            os.remove(active_file)

            # Using active file to get the orginal model back
            decrypted_model = 'decrypted_worker_model_'+str(node_num)
            m = pickle.load(open(decrypted_model, 'rb'))
            m.feature_names_in_ = np.array(['age', 'gender', 'height', 'weight', 'waist', 'hdl', 'ldl',
                                           'serum_creatinine', 'alt', 'gtp', 'dental_caries', 'tartar', 'triglyceride', 'hemoglobin'])

            # ask for input data file from user in csv format
            in_file_csv = input("Provide input dataset filepath: ")

            if get_consent():
                # load and process data
                x, y = preprocess_data(in_file_csv)

                # train model using partial_fit on new data and print accuracy for this trained model
                a = train_model(m, x, y, print_flag=True)
                print('Model trained on input dataset')

                # encrypt the model
                final_encrypted_model = 'worker_model_'+str(node_num)
                encrypt_model_parameters(a, final_encrypted_model, node_num)

                dest_file = 'master-node/child/'+final_encrypted_model
                store_files_s3('tenseal_encrypted_model_' +
                               str(node_num), dest_file)
                print('Homomorphic encrypted model uploaded to S3 bucket')
                print('\n****************************************\n')

            else:
                # print accuracy for this trained model
                print_test_accuracy(m, in_file_csv)

            # check for updated model files from master
            if s3_key_exists(passive_path):
                print('Aggregated model uploaded by master')

                # download passive master file
                get_files_s3(passive_path, passive_file, node_num)

                # remove current active file
                if os.path.isfile(active_file):
                    os.remove('active_worker_model_'+str(node_num))

                # replace active with passive
                os.rename(src=passive_file, dst=active_file)

                # delete s3 passive file
                delete_file_s3(passive_path)

                # upload local active file to s3 active
                delete_file_s3(active_path)
                store_files_s3(active_file, active_path)

        else:
            print("Waiting to receive model from master...")
            time.sleep(10)  # resume after 10 seconds


main()
