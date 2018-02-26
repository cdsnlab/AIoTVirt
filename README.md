# Chameleon: An Application-Aware Virtualization Architecture for Edge IoT Clouds


## Development environment

To clone the project, you need a key accepted by the Gitlab. If you don't have
one, please make one.

You are recommended to use JetBrains Pycharm for the development of Lapras
agents and contributions to the middleware.

### Python
Version: 3.5.3

Library: paho-mqtt, pymongo

## MQTT Message Format

### Service request (Example)
Topic: service/CriminalTracking

Message: {"name": "CriminalTracking", "type": "ObjectTracking", "requestTime": 12345, "parameter": {"object": "face", "resolution": 720}, "requirement": {"Performance": "HIGH", "Reliability": "MID"}}

## License

All rights are reserved by CDSN Lab, Korea Advanced Institute of Science and
Technology (KAIST).
