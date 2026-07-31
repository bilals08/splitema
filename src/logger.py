import logging

_FMT  = "%(asctime)s  %(levelname)-8s  %(message)s"
_DATE = "%H:%M:%S"


def get_logger(name: str = "transformer", level: int = logging.DEBUG) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
        log.addHandler(h)
        log.propagate = False
    log.setLevel(level)
    return log


log = get_logger()
