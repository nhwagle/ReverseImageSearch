import argparse
import pandas as pd
import os
from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors
import numpy as np
import subprocess
import sys

from sagemaker_containers.beta.framework import worker, encoders
from six import BytesIO

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

install('s3fs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. 
    parser.add_argument('--n_neighbors', type=int, default=10)
    #parser.add_argument('--metric', type=str, default='cosine')

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default='s3://narayani1/reverseimagesearch/knn_output')
    parser.add_argument('--model-dir', type=str, default='s3://narayani1/reverseimagesearch/model')
    parser.add_argument('--train', type=str, default='s3://narayani1/reverseimagesearch/points/points.csv')
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.train,header=None)
    points = df.values

    # Supply the hyperparameters of the nearest neighbors model
    n_neighbors = args.n_neighbors
    #metric = args.metric

    # Now, fit the nearest neighbors model
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
    model_nn = nn.fit(points)
    print('model has been fitted')

    # Save the model to the output location in S3
    joblib.dump(model_nn, os.path.join(args.model_dir, "model.joblib"))
    
    
def predict_fn(input_data, model):
    ind = model.kneighbors(input_data,return_distance=False)
    print('predict_fn output is ', ind[0])
    return ind[0]
    
    
def _npy_dumps(data):
    # Serializes a numpy array into a stream of npy-formatted bytes.
    buffer = BytesIO()
    np.save(buffer, data)
    return buffer.getvalue()

def output_fn(prediction_output, accept):
    if accept == 'application/x-npy':
        print('output_fn input is', prediction_output, 'in format', accept)
        return _npy_dumps(prediction_output), 'application/x-npy'
    elif accept == 'application/json':
        print('output_fn input is', prediction_output, 'in format', accept)
        return worker.Response(encoders.encode(prediction_output, accept), accept, mimetype=accept)
    else:
        raise ValueError('Accept header must be application/x-npy or application/json, but it is {}'.format(accept))
    
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

                          
                          
                          
                          
                          
                          
