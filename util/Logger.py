import logging
import logging.handlers

def Logger():
    # 로거 인스턴스를 만든다
    logger = logging.getLogger('mylogger')

    # 포매터를 만든다
    # formatter = logging.Formatter('[%(levelname)s|'+className+':%(lineno)s] %(asctime)s > %(message)s')
    formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

    # 스트림과 파일로 로그를 출력하는 핸들러를 각각 만든다.
    fileMaxByte = 1024 * 1024 * 100 #100MB
    fileHandler = logging.handlers.RotatingFileHandler('../Log.log', maxBytes=fileMaxByte, backupCount=10)
    streamHandler = logging.StreamHandler()

    # 각 핸들러에 포매터를 지정한다.
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    logger.setLevel(logging.DEBUG)

    return logger