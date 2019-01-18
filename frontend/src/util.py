import base64


def convert_image_to_base64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(bytearray(f.read()))
