import json
import logging.config
import os
from collections import OrderedDict

from tensorflow.python.lib.io import file_io

with file_io.FileIO(os.path.join(os.getcwd(), 'src/config.json'), 'r') as f:
    CONFS = json.load(f, object_pairs_hook=OrderedDict)

logging.config.dictConfig(CONFS['logger'])
LOGGER = logging.getLogger(CONFS['logger']['name'])
