import logging

lg = logging.getLogger(__name__)
lg.setLevel(logging.INFO)

formatter = logging.Formatter("%(message)s")

file_handler = logging.FileHandler("report.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.ERROR)

lg.addHandler(file_handler)
lg.addHandler(stream_handler)
