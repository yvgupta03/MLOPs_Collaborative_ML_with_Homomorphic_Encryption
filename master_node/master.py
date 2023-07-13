# imports
import os
import time

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier

from collections import Counter

import pickle
import tenseal as ts
import urllib3

import boto3
from botocore.exceptions import ClientError

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding


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


# function to perform data preprocessing
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
    y = df['smoking'].copy()
    x = df[['age', 'gender', 'height', 'weight', 'waist', 'hdl', 'ldl', 'serum_creatinine',
            'alt', 'gtp', 'dental_caries', 'tartar', 'triglyceride', 'hemoglobin']].copy()
    # normalizing x data to maintain the scale necessary for creation of model
    x = normalize_df(x)

    return x, y


# create base model from initial data
def create_master_model(x_train, y_train, print_flag=False):
    sgd = SGDClassifier()
    sgd.fit(x_train, y_train)

    if print_flag:
        x_train_prediction = sgd.predict(x_train)
        training_data_accuracy = accuracy_score(x_train_prediction, y_train)
        print('Training data accuracy: ', training_data_accuracy)

    return sgd


# function to get keys used for asymmetric encryption
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


# function that encrypts the model  file
def encrypt_model(model_pickle, encrypted_model):
    # get keys
    private_key, public_key = get_encryption_keys()

    # encrypt the model
    output = open(encrypted_model, 'ab')

    # perform encryption
    with open(model_pickle, 'rb') as input:
        while True:
            msg = input.read(100)

            if not msg:
                break

            encrypted = public_key.encrypt(
                msg,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            output.write(encrypted)

    output.close()


# function that decrypts the model file
def decrypt_worker_model(received_file, decrypted_file):
    # Decrypting the model
    input = open(decrypted_file, 'ab')

    # get keys
    private_key, public_key = get_encryption_keys()

    # perform decryption
    with open(received_file, 'rb') as output:
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


# decrypt all the received models
def decrypt_received_models():
    decrypt_worker_model('worker_model_1', 'decrypted_model_1')
    decrypt_worker_model('worker_model_2', 'decrypted_model_2')
    decrypt_worker_model('worker_model_3', 'decrypted_model_3')


# function to perform model aggregation
def perform_model_aggregation(print_flag=False):
    # define context for tenseal
    def context():
        context = ts.context(ts.SCHEME_TYPE.CKKS, 8192,
                             coeff_mod_bit_sizes=[60, 40, 40, 60])
        context.global_scale = pow(2, 40)
        context.generate_galois_keys()
        return context

    context = context()

    # fetch encrypted model parameters
    with open('worker_model_1', 'rb') as file:
        content1 = file.read()
    encrypted_params_model1 = ts.ckks_tensor_from(context, content1)

    with open('worker_model_2', 'rb') as file:
        content2 = file.read()
    encrypted_params_model2 = ts.ckks_tensor_from(context, content2)

    with open('worker_model_3', 'rb') as file:
        content3 = file.read()
    encrypted_params_model3 = ts.ckks_tensor_from(context, content3)

    # perform aggregation of model params
    mul = np.array([[0.33 for x in range(15)]])
    mul_tensor = ts.plain_tensor(mul)
    encrypted_params_aggregate = (
        encrypted_params_model1 + encrypted_params_model2 + encrypted_params_model3) * mul

    # decrypt the aggregated params
    decrypted = encrypted_params_aggregate.decrypt().tolist()
    sgd_params_aggregate = np.array(decrypted)

    # fit the parameters to the new model
    sgd_aggregate_intercept = np.array([sgd_params_aggregate[0][0]])
    sgd_aggregate_coef = np.reshape(sgd_params_aggregate[0][1:], (1, 14))

    x_train, y_train = preprocess_data(
        'https://personal.utdallas.edu/~pxn210006/dataset/dataset_initial_model.csv')

    aggregated_model = SGDClassifier()
    aggregated_model.partial_fit(x_train, y_train, np.array([0, 1]))
    aggregated_model.coef_ = sgd_aggregate_coef
    aggregated_model.intercept_ = sgd_aggregate_intercept
    aggregated_model.classes_ = np.array([0, 1])

    if print_flag:
        x_test, y_test = preprocess_data(
            'https://personal.utdallas.edu/~pxn210006/dataset/dataset_test.csv')
        score = aggregated_model.score(x_test, y_test)
        print('Model accuracy on test data', score)

    return aggregated_model


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


# function that sends files to s3 bucket
def store_files_s3(local_file_name, dest_file_name):
    s3_client = get_s3_client()
    bucketName = 'team-20-sptopic-master'

    dest1 = 'worker-node1/'+dest_file_name+'1'
    dest2 = 'worker-node2/'+dest_file_name+'2'
    dest3 = 'worker-node3/'+dest_file_name+'3'

    dest_par = 'master-node/parent/master_model'

    s3_client.upload_file(local_file_name, bucketName, dest1)
    s3_client.upload_file(local_file_name, bucketName, dest2)
    s3_client.upload_file(local_file_name, bucketName, dest3)

    s3_client.upload_file(local_file_name, bucketName, dest_par)


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
def get_files_s3(models, child_path):
    s3_client = get_s3_client()
    bucketName = 'team-20-sptopic-master'

    file1 = child_path + models[0]
    file2 = child_path + models[1]
    file3 = child_path + models[2]

    s3_client.download_file(bucketName, file1, models[0])
    s3_client.download_file(bucketName, file2, models[1])
    s3_client.download_file(bucketName, file3, models[2])


# function that deletes files from local directory and s3 bucket
def delete_files_local_s3(models, child_path):
    s3_client = get_s3_client()
    bucketName = 'team-20-sptopic-master'

    file1 = child_path + models[0]
    os.remove(models[0])
    s3_client.delete_object(Bucket=bucketName, Key=file1)

    file2 = child_path + models[1]
    os.remove(models[1])
    s3_client.delete_object(Bucket=bucketName, Key=file2)

    file3 = child_path + models[2]
    os.remove(models[2])
    s3_client.delete_object(Bucket=bucketName, Key=file3)


# main function
def main():

    # parent_path = 'master-node/parent/'
    child_path = 'master-node/child/'

    # STEP 1
    # create the master base model

    x_train, y_train = preprocess_data(
        'https://personal.utdallas.edu/~pxn210006/dataset/dataset_initial_model.csv')
    sgd_base_model = create_master_model(x_train, y_train, print_flag=True)
    # create pickle of the model
    pickle.dump(sgd_base_model, open('base_model_pickle', 'wb'))

    # now encrypt the base model
    encrypt_model('base_model_pickle', 'master_model')
    os.remove('base_model_pickle')

    # upload encrypted base model to s3 bucket
    store_files_s3('master_model', 'active_worker_model_')
    print('Base model trained and uploaded')

    # STEP 2
    # perform model aggregation

    while (True):
        # check for each model file if it exists
        flag = True

        worker_models_list = ['worker_model_1',
                              'worker_model_2', 'worker_model_3']

        for model in worker_models_list:
            filepath = child_path+model
            flag = s3_key_exists(filepath)

            if not flag:
                print('Waiting to receive model from all active clients!!!!')
                break

        # model file exists
        if flag:
            print('\n********************************************\n')
            print('Received model from all active clients!!!!')

            # download and save models
            get_files_s3(worker_models_list, child_path)
            print('All models downloaded. Now aggregation will start!!!!')

            # perform aggregation
            aggregated_model = perform_model_aggregation(True)
            print('Aggregation successfull')

            # delete old models
            delete_files_local_s3(worker_models_list, child_path)
            print('Delete old files from child directory')

            # create pickle of the model
            pickle.dump(aggregated_model, open('aggregate_model_pickle', 'wb'))

            # now encrypt the base model
            os.remove('master_model')
            encrypt_model('aggregate_model_pickle', 'master_model')
            os.remove('aggregate_model_pickle')

            # upload encrypted base model to s3 bucket
            store_files_s3('master_model', 'passive_master_model')
            print('New aggregated model trained and uploaded')
            print('\n******************************************\n')

        time.sleep(10)


main()
