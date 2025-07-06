import pytest
from logging_config import get_logger

def test_secret_redaction(tmp_path, caplog):
    logger = get_logger('testlogger')
    caplog.set_level('INFO')
    msg = 'User logged in with token=supersecret123 and password=hunter2'
    logger.info(msg)
    logs = caplog.text
    assert 'supersecret123' not in logs
    assert 'hunter2' not in logs
    assert 'token=***' in logs
    assert 'password=***' in logs

def test_log_rotation(tmp_path):
    logger = get_logger('rotationtest')
    for _ in range(10000):
        logger.info('Filling logs for rotation testing.')
    # Check that log file exists and is not over-rotated
    import os
    log_dir = 'logs'
    files = os.listdir(log_dir)
    assert any(f.startswith('pray.log') for f in files)
    assert len([f for f in files if f.startswith('pray.log')]) <= 6

