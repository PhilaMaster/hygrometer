import argparse
import redis
import time
from datetime import datetime
import threading
import adafruit_dht
import uuid
from board import D4
import numpy as np
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import sounddevice as sd
import torch


class SmartHygrometer:
    def __init__(self, redis_host, redis_port, redis_user, redis_pass):
        #DHT-11 initializing
        self.mac_address = hex(uuid.getnode())
        self.dht_device = adafruit_dht.DHT11(D4)

        #redis connection
        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port,
            username=redis_user, password=redis_pass
        )
        self._init_redis()
        print('Redis Connected:', self.redis_client.ping())

        #whisper model loading
        model_name = 'openai/whisper-tiny.en'
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

        #audio parameters
        self.AUDIO_DEVICE = 1
        self.CHANNELS = 1
        self.BIT_DEPTH = 'int16'
        self.SAMPLING_RATE = 48000     #48k khz
        self.WINDOW_SEC = 1            #1 second windows

        self.data_collection_enabled = False
        self.buffer_lock = threading.Lock()#lock to avoid race conditions on the audio buffer
        self.audio_buffer = np.zeros(self.SAMPLING_RATE * self.WINDOW_SEC, dtype=np.int16)
        self.last_data_time = 0

    def _init_redis(self):
        try:
            #create timeseries if not yet existing
            self.redis_client.ts().create('temp_timeseries')
            self.redis_client.ts().create('hum_timeseries')
        except redis.ResponseError:
            pass

    def audio_callback(self, indata, frames, callback_time, status):
        with self.buffer_lock:
            self.audio_buffer = indata.copy().flatten()

    def recognize_command(self):
        with self.buffer_lock:
            local_buffer = self.audio_buffer.copy()

        #conversion, normalization and downsampling
        waveform = local_buffer.astype(np.float32) / 32768.0
        waveform = waveform[np.newaxis, :]
        waveform16k = torchaudio.functional.resample(
            torch.tensor(waveform), self.SAMPLING_RATE, 16000
        ).squeeze()
        input_features = self.processor(
            waveform16k, sampling_rate=16000, return_tensors="pt"
        ).input_features
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        cmd = "".join(c for c in transcription if c.isalnum())
        return cmd.lower()

    def collect_and_send_data(self):
        timestamp = time.time()
        formatted_datetime = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
        try:
            temperature = self.dht_device.temperature
            humidity = self.dht_device.humidity
            print(f'{formatted_datetime} - {self.mac_address}:temperature = {temperature}')
            print(f'{formatted_datetime} - {self.mac_address}:humidity = {humidity}')
            timestamp_ms = int(timestamp * 1000)
            self.redis_client.ts().add('temp_timeseries', timestamp_ms, temperature)
            self.redis_client.ts().add('hum_timeseries', timestamp_ms, humidity)
        except:
            print(f'{formatted_datetime} - sensor failure')
            self.dht_device.exit()
            self.dht_device = adafruit_dht.DHT11(D4)

    def run(self):
        print("Sistema Smart Hygrometer avviato. Pronuncia 'up' per abilitare, 'stop' per disabilitare.")
        with sd.InputStream(device=self.AUDIO_DEVICE,
                            channels=self.CHANNELS,
                            dtype=self.BIT_DEPTH,
                            samplerate=self.SAMPLING_RATE,
                            blocksize=self.SAMPLING_RATE * self.WINDOW_SEC,
                            callback=self.audio_callback):
            while True:
                time.sleep(self.WINDOW_SEC)
                print("[VOICE] Begin audio recording...")
                cmd = self.recognize_command()
                if 'up' in cmd:
                    self.data_collection_enabled = True
                    print('[VOICE] Data collection ENABLED')
                elif 'stop' in cmd:
                    self.data_collection_enabled = False
                    print('[VOICE] Data collection DISABLED')

                cur_time = time.time()
                if self.data_collection_enabled and cur_time - self.last_data_time >= 5:
                    self.collect_and_send_data()
                    self.last_data_time = cur_time


def main():
    #parsing redis arguments
    parser = argparse.ArgumentParser(description='Smart Hygrometer with VUI & Redis')
    parser.add_argument('--host', type=str, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--user', type=str, required=True)
    parser.add_argument('--password', type=str, required=True)
    args = parser.parse_args()

    hygrometer = SmartHygrometer(
        redis_host=args.host,
        redis_port=args.port,
        redis_user=args.user,
        redis_pass=args.password
    )
    hygrometer.run()


if __name__ == "__main__":
    main()
