import logging


class ExtraDefaultFilter(logging.Filter):
    def __init__(self, filter_name):
        super(ExtraDefaultFilter, self).__init__(filter_name)

    def filter(self, record):
        if not hasattr(record, "clazz"):
            record.clazz = "Unknown"

        if not hasattr(record, "oid"):
            record.oid = "-"
        return True
