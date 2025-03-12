import logging

from Environment.helper.logging.simulation_context_filter import SimulationContextFilter

logging.basicConfig(format='%(asctime)s [%(levelname)s] [Time %(env_time)10s] [%(clazz)30s %(oid)3s]: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

lg = logging.getLogger()
lg.setLevel(logging.INFO)

logging.getLogger("fiona").setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] [Time %(env_time)10s] [%(clazz)30s %(oid)3s]: %(message)s')
file_handler = logging.FileHandler("report.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

lg.addHandler(file_handler)
lg.addHandler(stream_handler)

cf_init = SimulationContextFilter(filter_name='add_env', extra=[])
lg.addFilter(cf_init)
