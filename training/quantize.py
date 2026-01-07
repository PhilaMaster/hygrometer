import numpy as np
import onnx
import onnxruntime as ort
import os
import random
import torch

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    StaticQuantConfig,
    quantize,
)

from manu_training.DS_CNN import DSCNN
from msc_dataset import MSCDataset


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

CLASSES = ['stop', 'up']

calibration_ds = MSCDataset('../msc_val/', CLASSES)
test_ds = MSCDataset('../msc_test/', CLASSES)

MODEL_NAME = 'manu_model_96'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = DSCNN(num_classes=2, in_channels=1)
# model.load_state_dict(torch.load('best_model.pth', map_location=device))
# model.to(device)




frontend_float32_file = f'{MODEL_NAME}_frontend.onnx'
model_float32_file = f'{MODEL_NAME}_model.onnx'
ort_frontend = ort.InferenceSession(frontend_float32_file)
ort_model = ort.InferenceSession(model_float32_file)

true_count = 0.0
for sample in test_ds:
    inputs = sample['x']
    label = sample['y']
    inputs = inputs.numpy()
    inputs = np.expand_dims(inputs, 0)
    features = ort_frontend.run(None, {'input': inputs})[0]
    outputs = ort_model.run(None,  {'input': features})[0]
    prediction = np.argmax(outputs, axis=-1).item()
    true_count += prediction == label

float32_accuracy = true_count / len(test_ds) * 100
frontend_size = os.path.getsize(frontend_float32_file)
model_float32_size = os.path.getsize(model_float32_file)
total_float32_size = frontend_size + model_float32_size

print(f'Float32 Accuracy: {float32_accuracy:.2f}%')
print(f'Float32 Frontend Size: {frontend_size / 2**10:.1f}KB')
print(f'Float32 Model Size: {model_float32_size / 2**10:.1f}KB')
print(f'Float32 Total Size: {total_float32_size / 2**10:.1f}KB')

class DataReader(CalibrationDataReader):
    def __init__(self, dataset):
        self.dataset = dataset
        self.enum_data = None

        self.datasize = len(self.dataset)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.dataset)

        x = next(self.enum_data, None)

        if x is None:
            return None

        x = x['x']
        x = x.numpy()
        x = np.expand_dims(x, 0)
        x = ort_frontend.run(None, {'input': x})[0]
        x = {'input': x}

        return x

    def rewind(self):
        self.enum_data = None


data_reader = DataReader(calibration_ds)

conf = StaticQuantConfig(
    calibration_data_reader=data_reader,
    quant_format=QuantFormat.QDQ,
    calibrate_method=CalibrationMethod.MinMax ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=False,
)

model_int8_file = f'quantized_models/{MODEL_NAME}_INT8.onnx'
quantize(model_float32_file, model_int8_file, conf)

ort_model_int8 = ort.InferenceSession(model_int8_file)

true_quant_count = 0.0
for sample in test_ds:
    inputs = sample['x']
    label = sample['y']
    inputs = inputs.numpy()
    inputs = np.expand_dims(inputs, 0)
    features = ort_frontend.run(None, {'input': inputs})[0]
    outputs = ort_model_int8.run(None,  {'input': features})[0]
    prediction = np.argmax(outputs, axis=-1).item()
    true_quant_count += prediction == label

int8_accuracy = true_quant_count / len(test_ds) * 100
frontend_size = os.path.getsize(frontend_float32_file)
model_int8_size = os.path.getsize(model_int8_file)
total_int8_size = frontend_size + model_int8_size

print(f'INT8 Accuracy: {int8_accuracy:.2f}%')
print(f'Float32 Frontend Size: {frontend_size / 2**10:.1f}KB')
print(f'INT8 Model Size: {model_int8_size / 2**10:.1f}KB')
print(f'INT8 Total Size: {total_int8_size / 2**10:.1f}KB')