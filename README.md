<img width="993" height="286" alt="image" src="https://github.com/user-attachments/assets/c49b2204-7c87-4ddb-b190-4cf7ea570b31" />

This project implements a smart hygrometer running on a Raspberry Pi equipped with a DHT-11 sensor and a USB microphone. The device periodically measures temperature and humidity, sends measurements to a Redis Cloud database, and uses a custom Keyword Spotting (KWS) pipeline to start or stop data collection through voice commands ("up" and "stop").

## Features

- **Efficient KWS Pipeline**: Custom-trained MobileNet-inspired DS-CNN model achieving 99.5% accuracy with <300 KB total ONNX size and <5 ms latency
- **Voice-Controlled Interface**: Real-time keyword detection enabling hands-free control of data collection
- **Periodic Sensor Acquisition**: Temperature and humidity measurement from DHT-11 sensor every 5 seconds when enabled
- **Cloud Storage**: Time-series data storage in Redis Cloud database for remote monitoring and visualization
- **Resource-Constrained Optimization**: Optimized for deployment on Raspberry Pi with MFCC feature extraction and int8 quantization

## Model Architecture

The KWS pipeline consists of two ONNX modules:

### Feature Extraction
- **MFCC-based preprocessing** with optimized hyperparameters:
  - Frame length: 32 ms, Frame step: 24 ms
  - 10 mel-filterbanks, 8 MFCCs
  - Frequency range: 20-8000 Hz

### DS-CNN Classifier
- **MobileNet-inspired architecture** with depthwise separable convolutions
- 96 channels across 4 DS-Conv blocks (3×3 depthwise + 1×1 pointwise)
- Global Average Pooling + Dropout (0.3) + FC layer
- **Post-training quantization**: float32 → int8 (64% size reduction)

### Training Configuration
- 2000 training steps with batch size 32
- Adam optimizer (lr=0.001) with StepLR scheduler (step=800, gamma=0.1)
- CrossEntropy loss function

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 99.5% |
| Total ONNX Size | 270.7 KB |
| Median Latency (RPI) | 3.5 ms |

All constraints exceeded: accuracy >99.4%, size <300 KB, latency <5 ms.

## System Requirements

- Raspberry Pi (tested on RPI 4)
- DHT-11 temperature/humidity sensor
- USB microphone
- Redis Cloud account

## Configuration

Set Redis credentials via command-line arguments:

- `--host`: Redis Cloud host
- `--port`: Redis Cloud port
- `--user`: Redis Cloud username
- `--password`: Redis Cloud password

### Example Usage

```bash
python hygrometer.py \
  --host <REDIS_HOST> \
  --port <REDIS_PORT> \
  --user <REDIS_USER> \
  --password <REDIS_PASSWORD>
```

## Voice Command Logic

The system analyzes 1-second audio windows every second:

- **"up"** (probability >99.9%): Enable data collection
- **"stop"** (probability >99.9%): Disable data collection
- **Low confidence** (≤99.9%): Maintain current state

Audio preprocessing pipeline:
1. Convert int16 audio to float32 PyTorch tensor
2. Channel-last → channel-first layout
3. Normalize to [-1, 1]
4. Downsample 48 kHz → 16 kHz
5. Convert to numpy array with batch dimension
6. Feed to ONNX KWS pipeline
7. Softmax output to probability distribution

## Implementation Details

### Methodology
The KWS pipeline was developed through an iterative, constraint-driven approach:
1. Initial experiments revealed accuracy-latency-size trade-offs
2. Adopted MobileNet architecture for resource efficiency
3. Systematic grid search for optimal MFCC hyperparameters
4. Post-training quantization for final compression

### Key Optimizations
- **Depthwise separable convolutions** reduced parameter count vs. standard convolutions
- **MFCC feature extraction** (8 coefficients, 10 mel-filterbanks) minimizes computational overhead
- **96-channel configuration** optimally balances size constraints and accuracy
- **int8 quantization** with calibrated activation ranges preserves accuracy while reducing size by 64%
