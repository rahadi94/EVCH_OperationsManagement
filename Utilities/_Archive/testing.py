from pathlib import Path

import time

# TODO: We need to use configuration file and get rid of initialization


import sys
from configparser import ConfigParser
import os

# Change working directory to path of run.py
working_dir = Path(__file__).parent
os.chdir(working_dir)

# Read args
parser = ConfigParser()
parser.read(sys.argv[1])

start_time = time.time()

CHARGER_CAPA = parser.get("GRID", "ELECTRICITY_TARIFF").split(",")
for i in range(0, len(CHARGER_CAPA)):
    CHARGER_CAPA[i] = int(CHARGER_CAPA[i])
print(CHARGER_CAPA)
