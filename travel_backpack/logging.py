from travel_backpack.time import time_now_to_string
import functools


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


_log_wrapper_recursion = [None]


def log_wrapper(f):

    f_name = f.__code__.co_name

    @functools.wraps(f)
    def _log_wrapper(*args, **kwargs):

        if not _log_wrapper_recursion[0]:
            # f_name = f.__name__
            _log_wrapper_recursion[0] = f_name
            try:
                f_args = ', '.join(map(str, args))
                f_kwargs = ', '.join([f'{k}={v}' for k, v in kwargs.items()])
                f_cargs = f_args
                if f_kwargs:
                    f_cargs += ', ' + f_kwargs

                f_call = f'{f_name}({f_cargs})'
            except:
                f_call = f'{f_name}(?<exception>?)'
            _log_wrapper_recursion[0] = None
        else:
            f_call = f'{f_name}(?{_log_wrapper_recursion[0]}?)'

        print('=== [log][trace]', '>>>', f_call)

        try:
            result = f(*args, **kwargs)

            if not _log_wrapper_recursion[0]:
                _log_wrapper_recursion[0] = f_name
                try:
                    f_result = f'{result} from {f_call}'
                except:
                    f_result = f'(?<exception>?) from {f_call}'
                _log_wrapper_recursion[0] = None
            else:
                f_result = f'(?{_log_wrapper_recursion[0]}?) from {f_call}'
            print('=== [log][trace]', '<<<', f_result)
        except Exception as ex:
            print('=== [log][trace]', '<<<exception<<<', f'({type(ex)} {ex})', 'from', f_call)
            raise

        return result

    return _log_wrapper


def setup_tracer(name,
                 log_decorator=log_wrapper,
                 dont_log_filter=lambda name: name.startswith('__') and not name.endswith('__')):
    import travel_backpack
    import travel_backpack.decorators
    import sys
    names_to_ignore = [log_decorator.__code__.co_name, log_decorator(lambda x: x).__code__.co_name]

    def log(f):
        if getattr(f, '__code__', None) != None:
            f_name = f.__code__.co_name
            if f_name not in names_to_ignore and not dont_log_filter(f_name):
                # if hasattr(f, '__wrapped__'):
                #     f_child = f.__wrapped__
                #     wrapped_f_child = log(f_child)
                #     f.__wrapped__ = wrapped_f_child

                if travel_backpack.is_local_code(f, name=name):
                    # print('wrapping', f_name, f)
                    return log_decorator(f)

                else:
                    # print('not local', f)
                    return f

            else:
                # print('func already wrapped')
                return f
        else:
            # print('weird function')
            return f

    decorate_methods = travel_backpack.decorators.decorate_all_methods
    decorate_functions = travel_backpack.decorators.decorate_all_functions_in_module
    decorate_classes = travel_backpack.decorators.decorate_all_classes_in_module
    decorate_modules = travel_backpack.decorators.decorate_all_modules_in_module

    def hyper_logger_decorator(function_decorator, class_decorator):
        decorated_modules = set()

        def wrapper(module):
            if module not in decorated_modules:
                decorated_modules.add(module)
                decorate_functions(module, function_decorator)
                decorate_classes(module, class_decorator)
                decorate_modules(module, wrapper, name)
            # else:
            #     print('already decorated:', module)

            return module

        return wrapper

    hyper_logger_decorator(log, decorate_methods(log))(sys.modules[name])