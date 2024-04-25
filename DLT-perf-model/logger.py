import logging
import sys


def init_logging():
    # file_handler = logging.FileHandler(filename='tmp.log')
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [
        # file_handler,
        stdout_handler
    ]

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )
