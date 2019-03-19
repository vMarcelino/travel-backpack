"""travel-backpack - Some very useful functions and classes to use in day-to-day"""

__version__ = '0.4.0'
__author__ = 'Victor Marcelino <victor.fmarcelino@gmail.com>'
__all__ = []

time_to_string = lambda x: '%04d/%02d/%02d - %02d:%02d:%02d' % (x.year, x.month, x.day, x.hour, x.minute, x.second)


def time_now_to_string(separators=None, order=None, lengths=None):
    if order is None:
        order = ['y', 'mo', 'd', 'h', 'mi', 's']
    if separators is None:
        separators = ['/', '/', ' - ', ':', ':', '.']
    if lengths is None:
        lengths = [4, 2, 2, 2, 2, 2, 4]
    elif type(lengths) is int:
        lengths = [lengths] * len(order)

    if len(separators) < len(order) - 1:
        raise Exception('Separator count must be at least order count - 1')
    if len(lengths) < len(order):
        raise Exception('Lengths count must be at least order count or a single int')

    from datetime import datetime as dt
    x = dt.now()
    y = x.year
    m = x.month
    d = x.day
    h = x.hour
    mi = x.minute
    s = x.second
    us = x.microsecond

    var_map = {'y': y, 'mo': m, 'd': d, 'h': h, 'mi': mi, 's': s, 'u': us}
    result = ''
    for i, e in enumerate(order):
        if i > 0:
            result += separators[i - 1]
        result += f'{{0:0{lengths[i]}d}}'.format(var_map[e])
    return result


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

        return cls._instances[cls]


def log_info(msg, file, print_to_console=True, print_time=True):
    nmsg = '[' + time_now_to_string() + '] ' + msg
    if print_to_console:
        if print_time:
            print(nmsg)
        else:
            print(msg)
    with open(file, "a+") as f:
        f.write(nmsg + "\n")


from functools import wraps


def multi_replace(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


def pp(*args, **kwargs):
    from pprint import PrettyPrinter
    pp = PrettyPrinter(indent=4).pprint
    pp(*args, **kwargs)


def decorate_all_methods(decorator):
    def decorate(cls):
        for attr in cls.__dict__:  # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                print('setting', attr)
                setattr(cls, attr, decorator(getattr(cls, attr)))
            else:
                print('ignoring', attr)
        print(cls, type(cls), 'callable:', callable(cls))
        return cls

    return decorate


def format_exception_string(e):
    import traceback
    s = f'Error: {type(e)} {str(e)}\n'
    s += 'Traceback: ' + '\n'.join(traceback.format_list(traceback.extract_tb(e.__traceback__)))
    return s


def except_and_print(func, raise_exception=False):
    @wraps(func)
    def decorator(*args, **kwargs):
        try:
            return func(*args, **kwargs)

        except Exception as e:
            print(format_exception_string(e))
            if raise_exception:
                raise

    return decorator


def supports_color():
    """
    Returns True if the running system's terminal supports color, and False
    otherwise.
    """
    import os
    import sys
    plat = sys.platform
    supported_platform = plat != 'Pocket PC' and (plat != 'win32' or 'ANSICON' in os.environ)
    # isatty is not always implemented, #6223.
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    if not supported_platform or not is_a_tty:
        return False
    return True


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def copy(src, dst):
    import shutil
    make_folder(dst, is_file=True)
    shutil.copy2(src, dst)
    return


def make_folder(path, is_file=False):
    import os
    if is_file:
        dirname = os.path.dirname(path)
        if dirname != '':
            make_folder(dirname)
    else:
        try:
            os.stat(path)
        except:
            os.makedirs(path)


def remove_folder(path):
    import shutil
    shutil.rmtree(path)


def file_exists(path):
    import os
    try:
        os.stat(path)
        return True
    except Exception as ex:
        return False


def dir_exists(path):
    return file_exists(path)


def check_var_input(message: str, end: str = '\n -> '):
    while True:
        v = input(message + end)
        if binary_user_question(f'is {v} correct?'):
            return v


def binary_user_question(message: str,
                         end: str = '\n -> ',
                         error_message: str = 'Invalid answer',
                         true_message: str = 'y',
                         false_message: str = 'n',
                         default: bool = True,
                         exact: bool = False,
                         case_sensitive: bool = False):

    y_n_question = f'[{true_message}]/{false_message}' if default is True else f'{true_message}/[{false_message}]'

    while True:
        ans = input(f'{message} {y_n_question}{end}')
        if case_sensitive:
            ans, true_message, false_message = map(lambda s: s.lower(), (ans, true_message, false_message))
        if exact:
            if ans == true_message:
                return True
            elif ans == false_message:
                return False
        else:
            return ans != false_message if default is True else ans == true_message

        print(error_message)


def format_date_to_path(date):
    return date.replace('/', '-').replace(':', 'h').replace('_', ' ')


class Logger(object):
    def __init__(self, logpath=None, logpath_last=None, timestamp_log=True, timestamp_terminal=True, timestamp_func=None):
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
    sys.stdout = Logger(log_file)


_DEFAULT_POOL = None


def threadpool(f, executor=None):
    global _DEFAULT_POOL
    import concurrent.futures
    if _DEFAULT_POOL is None:
        _DEFAULT_POOL = concurrent.futures.ThreadPoolExecutor()

    @wraps(f)
    def wrap(*args, **kwargs):
        return (executor or _DEFAULT_POOL).submit(f, *args, **kwargs)

    return wrap


def thread_encapsulation(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        from threading import Thread
        t = Thread(target=f, args=args, kwargs=kwargs)
        t.start()
        return t

    return decorator


def print_function_name(func_or_arg="", show_function_call=True, show_function_return=False):
    prefix = ""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if func is None:
                if show_function_call or show_function_return: print('tried to execute unnamed (null) function')
                raise Exception()
            else:
                func_name = prefix + func.__name__
                kwargs_str_list = [f"{k}={v}" for k, v in kwargs.items()]
                args_str_list = list(map(str, args))
                if show_function_call: print(f'--> {func_name}({", ".join(args_str_list + kwargs_str_list)})')
                r = func(*args, **kwargs)
                if show_function_return: print(f'<-- {func_name} returned {r}')
                return r

        return wrapper

    if callable(func_or_arg):
        return decorator(func_or_arg)

    else:
        prefix = func_or_arg
        return decorator
