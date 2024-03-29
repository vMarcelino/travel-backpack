"""travel-backpack - Some very useful functions and classes to use in day-to-day"""

__version__ = '3.4.1'
__author__ = 'Victor Marcelino <victor.fmarcelino@gmail.com>'
__all__ = []

from functools import wraps
import inspect
from typing import Sequence


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

        return cls._instances[cls]


def multi_replace(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


def pp(*args, **kwargs):
    from pprint import PrettyPrinter
    pp = PrettyPrinter(indent=4).pprint
    pp(*args, **kwargs)


def is_local_code(obj, *, module=None, path: str = None, name: str = None) -> bool:
    import os
    import sys

    s = sum([module is not None, path is not None, name is not None])

    if s == 0:
        raise Exception('module, file or name must be given')
    elif s > 1:
        raise Exception('only one of the parameters can be used at the same time')
    else:
        if module:
            return is_local_code(obj, path=inspect.getabsfile(module))

        elif name:
            return is_local_code(obj, module=sys.modules[name])

        elif path:
            topmost_dir = os.path.dirname(os.path.realpath(path))

            if any([inspect.isfunction(obj), inspect.ismodule(obj), inspect.ismethod(obj)]):
                module_path = os.path.realpath(inspect.getabsfile(obj))
                return topmost_dir in module_path

            else:
                raise NotImplementedError

        else:
            raise Exception


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
    ENDC = '\x1b[0m'

    # some commonly used
    HEADER = '\x1b[95m'
    OKBLUE = '\x1b[94m'
    OKGREEN = '\x1b[92m'
    WARNING = '\x1b[93m'
    REDHIGHLIGHT = '\x1b[41m'
    FAIL = '\x1b[91m'

    # font attributes
    BOLD = '\x1b[1m'
    BOLD_OFF = '\x1b[21m'
    UNDERLINE = '\x1b[4m'
    UNDERLINE_OFF = '\x1b[24m'
    BLINK = '\x1b[5m'
    BLINK_OFF = '\x1b[21m'

    # foreground colors
    BLACK = '\x1b[30m'
    RED = '\x1b[31m'
    GREEN = '\x1b[32m'
    YELLOW = '\x1b[33m'
    BLUE = '\x1b[34m'
    MAGENTA = '\x1b[35m'
    CYAN = '\x1b[36m'
    WHITE = '\x1b[37m'

    DEFAULT = '\x1b[39m'

    GRAY_LIGHT = '\x1b[90m'
    RED_LIGHT = '\x1b[91m'
    GREEN_LIGHT = '\x1b[92m'
    YELLOW_LIGHT = '\x1b[93m'
    BLUE_LIGHT = '\x1b[94m'
    MAGENTA_LIGHT = '\x1b[95m'
    CYAN_LIGHT = '\x1b[96m'
    WHITE_LIGHT = '\x1b[97m'

    # background colors
    BGBLACK = '\x1b[40m'
    BGRED = '\x1b[41m'
    BGGREEN = '\x1b[42m'
    BGYELLOW = '\x1b[43m'
    BGBLUE = '\x1b[44m'
    BGMAGENTA = '\x1b[45m'
    BGCYAN = '\x1b[46m'
    BGWHITE = '\x1b[47m'

    BGDEFAULT = '\x1b[49m'

    BGGRAY_LIGHT = '\x1b[100m'
    BGRED_LIGHT = '\x1b[101m'
    BGGREEN_LIGHT = '\x1b[102m'
    BGYELLOW_LIGHT = '\x1b[103m'
    BGBLUE_LIGHT = '\x1b[104m'
    BGMAGENTA_LIGHT = '\x1b[105m'
    BGCYAN_LIGHT = '\x1b[106m'
    BGWHITE_LIGHT = '\x1b[107m'


def colorize(text: str, color: str) -> str:
    '''
    Colorizes the input text with ANSI colors
    '''
    return color + text + bcolors.ENDC


def table_print(arr: 'Sequence[Sequence[str]]', colors:'tuple[str,...]'=(bcolors.WHITE, bcolors.GRAY_LIGHT)):
    line_count = len(arr)
    col_count = len(arr[0])
    col_size = [0] * col_count
    for i in range(col_count):
        for j in range(line_count):
            col_size[i] = max(col_size[i], len(arr[j][i]))

    for j, line in enumerate(arr):
        color = colors[j % len(colors)]
        for i, item in enumerate(line):
            print(colorize(item.rjust(col_size[i], ' '), color), end='  |  ')
            if i == col_count - 1:
                print()


def copy(src, dst, dst_is_file=True):
    import shutil
    make_folder(dst, is_file=dst_is_file)
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
    except:
        return False


def dir_exists(path):
    return file_exists(path)


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
