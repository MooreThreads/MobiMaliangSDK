import logging
from logging.handlers import TimedRotatingFileHandler
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(__dir__))

logdir = "logdirs"

class Logger():
    def __init__(self, logger_name='server_logger', logdir=logdir):
        
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logdir = logdir
        os.makedirs(self.logdir,exist_ok=True)
        self.logfilename = os.path.join(self.logdir, 'server.log')
        
        
        self.consoleHandler = logging.StreamHandler()
        self.consoleHandler.setLevel(logging.WARNING)

        self.fileHandler = TimedRotatingFileHandler(filename=self.logfilename, when="D", backupCount=7, encoding="utf-8")
        self.fileHandler.setLevel(logging.INFO)
        
        self.formatter = logging.Formatter("%(asctime)s - %(filename)s ->\t%(funcName)s[line:%(lineno)d] - %(levelname)s: %(message)s")

        self.consoleHandler.setFormatter(self.formatter)
        self.fileHandler.setFormatter(self.formatter)

        self.logger.addHandler(self.consoleHandler)
        self.logger.addHandler(self.fileHandler)
        
    def getlog(self):
        
        self.logger.info("logger is loaded sucessfully!")
        return self.logger

server_logger = Logger(logdir=logdir).getlog()