# logging_config.py
import logging
import logging.config
import re
import os

SECRET_REGEX = re.compile(r'(token|password|secret)\s*=\s*[^\s,;]+', re.IGNORECASE)

class SecretRedactor(logging.Filter):
    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = SECRET_REGEX.sub(lambda m: f'{m.group(1)}=***', record.msg)
        return True

def get_logger(name=None):
    logger = logging.getLogger(name)
    if not logger.handlers:
        if not os.path.exists('logs'):
            os.makedirs('logs')
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,
            'filters': {'redact': {'()': __name__ + '.SecretRedactor'}},
            'formatters': {
                'detailed': {
                    'format': '[%(asctime)s] %(levelname)s %(name)s: %(message)s'
                }
            },
            'handlers': {
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': 'logs/pray.log',
                    'maxBytes': 1024*1024*10,  # 10MB
                    'backupCount': 5,
                    'formatter': 'detailed',
                    'level': 'DEBUG',
                    'filters': ['redact']
                },
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'detailed',
                    'level': 'INFO',
                    'filters': ['redact']
                }
            },
            'loggers': {
                '': {
                    'handlers': ['file', 'console'],
                    'level': 'DEBUG',
                    'propagate': True
                },
                'audit': {
                    'handlers': ['file', 'console'],
                    'level': 'INFO',
                    'propagate': False
                },
            }
        })
    logger.addFilter(SecretRedactor())
    return logger

