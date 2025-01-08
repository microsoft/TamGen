import base64


CACHE = {}

def load_data(input_data: dict):
    for k, v in input_data.items():
        CACHE[k] = base64.b64decode(v)
