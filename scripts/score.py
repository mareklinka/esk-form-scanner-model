# This script generates the scoring and schema files
# necessary to operationalize your model
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema

from keras.models import load_model as load_keras_model
from PIL import Image

import numpy as np

import os

# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.

model = None

def init():
    local_path = '.azureml/share/model.h5'

    global model
    model = load_keras_model(local_path)

def run(input_array):
    import json
    import base64
    import io

    img = Image.open(io.BytesIO(base64.b64decode(input_array)))
    data = np.array(img)
    data = data / 255
    data = np.expand_dims(data, axis=0)

    prediction = model.predict(data)
    return { "coordinates": prediction.tolist() }

def generate_api_schema():
    inputs = {"input_array": SampleDefinition(DataTypes.STANDARD, "dGVzdA==")}
    outputs = {"coordinates": SampleDefinition(DataTypes.STANDARD, [1, 2, 4, 5, 8, 9, 5, 6])}
    print(generate_schema(inputs=inputs, filepath="schema.json", run_func=run, outputs=outputs))

# generate_api_schema()

# import base64

# init()
# with open("E:\\Repos\\esk-form-scanner-model\\src\\data\\validation\\i0.jpg", "rb") as file:
#     print(base64.b64encode(file.read(-1)))