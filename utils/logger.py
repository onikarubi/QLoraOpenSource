from logging import INFO, Formatter, StreamHandler, getLogger


def get_logger(name: str, level: int = INFO):
    logger = getLogger(name)
    handler = StreamHandler()
    handler.setLevel(level)
    formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

