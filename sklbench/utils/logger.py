import logging

logger = logging.Logger("sklbench")

logging_channel = logging.StreamHandler()
logging_formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
logging_channel.setFormatter(logging_formatter)

logger.addHandler(logging_channel)
