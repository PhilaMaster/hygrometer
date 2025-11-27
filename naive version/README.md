<img width="1003" height="306" alt="image" src="https://github.com/user-attachments/assets/68c25933-4dc0-48a3-9cc6-a312d7452f51" />
This project implements a simplified smart hygrometer running on a Raspberry Pi equipped with a DHT‑11 sensor and a USB microphone.​
The device periodically measures temperature and humidity, sends measurements to a Redis Cloud database, and uses a small speech‑to‑text model to start or stop data collection through voice commands such as “up” and “stop”.​

## Features

- Periodic acquisition of temperature and humidity from a DHT‑11 sensor on Raspberry Pi.​

- Storage of measurements in Redis using time‑series style keys for later visualization.​

- Continuous audio capture from a USB microphone with fixed recording parameters (mono, 16‑bit, 48 kHz).​

- Keyword‑based control logic using a Whisper tiny‑style speech recognizer (data collection toggled by “up” / “stop”).​

## Configuration
Set Redis credentials either via command‑line arguments or environment variables:

--host: Redis Cloud host.​

--port: Redis Cloud port.

--user: Redis Cloud username.

--password: Redis Cloud password.

### Example:
```
python hygrometer.py \
  --host <REDIS_HOST> \
  --port <REDIS_PORT> \
  --user <REDIS_USER> \
  --password <REDIS_PASSWORD>
```
