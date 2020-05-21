from functools import wraps
from typing import Type, Union, TYPE_CHECKING, cast


def format_exception_string(e):
    import traceback
    s = f'Error: {type(e)} {str(e)}\n'
    s += 'Traceback: ' + '\n'.join(traceback.format_list(traceback.extract_tb(e.__traceback__)))
    return s


def except_and_print(func, message=None, raise_exception=False):
    @wraps(func)
    def decorator(*args, **kwargs):
        try:
            return func(*args, **kwargs)

        except Exception as e:
            if message is not None:
                print(message)
            print(format_exception_string(e))
            if raise_exception:
                raise

    return decorator


def except_and_print_with_message(message, *args, **kwargs):
    def eaprint(func, *args, **kwargs):
        return except_and_print(func, *args, message=message, **kwargs)

    return eaprint


def check_and_raise(condition: bool, exception: Union[Type[BaseException], BaseException] = Exception):
    """Raises an exception when condition is not met
    
    Arguments:
        condition {bool} -- condition to check against
    
    Keyword Arguments:
        exception {Union[Type[Exception],Exception]} -- the exception type to be raised (default: {Exception})
    
    Raises:
        exception_type: the exception of type exception_type, given in the parameter
    """
    if not condition:
        if TYPE_CHECKING:
            if isinstance(exception, BaseException):
                exception = cast(BaseException, exception)
                raise exception
            else:
                exception = cast(Type[BaseException], exception)
                raise exception
        else:
            raise exception
