from travel_backpack.time import time_now_to_string

def log_info(msg, file, print_to_console=True, print_time=True):
    nmsg = '[' + time_now_to_string() + '] ' + msg
    if print_to_console:
        if print_time:
            print(nmsg)
        else:
            print(msg)
    with open(file, "a+") as f:
        f.write(nmsg + "\n")


def multi_replace(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

class Logger(object):
    def __init__(self,
                 logpath=None,
                 logpath_last=None,
                 timestamp_log=True,
                 timestamp_terminal=True,
                 timestamp_func=None):
        import os
        import sys
        if timestamp_func is None:
            import time
            timestamp_func = lambda: f'[{time.strftime("%d/%m/%Y")} at {time.strftime("%H:%M:%S")}] '
            timestamp_func = lambda: f"[{time_now_to_string(separators=['/','/',' at ',':',':'])}] "

        if logpath is None:
            logpath = os.path.join(os.environ['userprofile'], os.path.basename(__file__) + '.log')

        if logpath_last is None:
            logpath_last = '.last_run'.join(os.path.splitext(logpath))

        self.terminal = sys.stdout
        self.log = open(logpath, "a+", 1)
        self.log_last = open(logpath_last, "w", 1)
        self.timestamp_log = timestamp_log
        self.timestamp_terminal = timestamp_terminal
        self.timestamp_func = timestamp_func

    def write(self, message):
        # Write to terminal
        if self.terminal:
            to_write_terminal = message
            if self.timestamp_terminal:
                to_write_terminal = to_write_terminal.replace('\n', f'\n{self.timestamp_func()}')

            self.terminal.write(to_write_terminal)

        # Write to log
        to_write_log = message
        if self.timestamp_log:
            to_write_log = to_write_log.replace('\n', f'\n{self.timestamp_func()}')

        self.log.write(to_write_log)
        self.log_last.write(to_write_log)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        self.log.flush()
        pass


def log_stdout(log_file):
    import sys
    log = Logger(log_file)
    sys.stdout = log
    sys.stderr = log
