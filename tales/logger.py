import datetime
import logging
import os
import platform
import re
from os.path import join as pjoin

from tqdm import tqdm

log = logging.getLogger("tales")


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class StripAnsiFormatter(logging.Formatter):
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")

    def format(self, record):
        msg = super().format(record)
        return self.ansi_escape.sub("", msg)


def setup_logging(args):
    log.setLevel(logging.DEBUG)

    def add_new_file_handler(logfile):
        fh = logging.FileHandler(logfile, mode="w")
        formatter = StripAnsiFormatter("%(asctime)s: %(message)s")
        log.addHandler(fh)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # Log some system information at the top of the log file.
        def _emit_msg(msg):
            fh.emit(
                logging.makeLogRecord(
                    {"name": log.name, "level": logging.DEBUG, "msg": msg}
                )
            )

        _emit_msg("System information:")
        _emit_msg(f"args = {args}")
        _emit_msg(f"system = {platform.system()}")
        _emit_msg(f"server = {platform.uname()[1]}")
        _emit_msg(f"working_dir = {os.getcwd()}")
        _emit_msg(f"datetime = {datetime.datetime.now()}")
        _emit_msg(f"git_commit = {os.popen('git rev-parse HEAD').read().strip()}")

        return fh

    log.add_new_file_handler = add_new_file_handler

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = pjoin(args.log_dir, f"{timestamp}.log")
    log.add_new_file_handler(logfile)

    ch = TqdmLoggingHandler()
    formatter = logging.Formatter("%(message)s")
    ch.setLevel(args.logging_level)
    ch.setFormatter(formatter)
    log.addHandler(ch)
