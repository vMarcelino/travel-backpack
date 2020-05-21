from dataclasses import dataclass
from travel_backpack.exceptions import check_and_raise
from typing import Any, Optional, Sequence, TypeVar, Callable, Type, Union, Tuple, cast, overload, NoReturn
T = TypeVar('T')


@dataclass
class VariableReferenceHolder:
    """Holds a reference to a variable

    Useful for passing as argument just like a
    variable pointer or a variable reference in other languages
    """
    value: Any


class Undefined:
    pass


@overload
def ensure_type(obj: Any,
                _type: Type[T],
                output_type: None = None,
                exception: Union[BaseException, Type[BaseException]] = TypeError) -> T:
    ...
@overload
def ensure_type(obj: Any,
                _type: Type[T],
                output_type: Optional[Type] = None,
                exception: Union[BaseException, Type[BaseException]] = TypeError) -> T:
    ...

@overload
def ensure_type(obj: Any,
                _type:Union[Type, Tuple[Type, ...]],
                output_type: Type[T],
                exception: Union[BaseException, Type[BaseException]] = TypeError) -> T:
    ...


def ensure_type(obj: Any,
                _type: Union[Type[T], Union[Type, Tuple[Type, ...]]],
                output_type: Optional[Type[T]] = None,
                exception: Union[BaseException, Type[BaseException]] = TypeError) -> T:
    # if output_type is None:
    #     if isinstance(_type, Tuple):
    #         raise ValueError('output_type must be set when more than one type is given')
    #     else:
    #         output_type = _type

    if isinstance(exception, BaseException):  # is instance of exception
        check_and_raise(isinstance(obj, _type), exception=exception)
    else:  # is class
        exception_message = f"Object of type {type(obj)} does not match type {_type}"
        check_and_raise(isinstance(obj, _type), exception=exception(exception_message))

    return cast(T, obj)


def check_var_input(output_type: Type[T],
                    message: str,
                    type_error_message: str = 'Invalid type',
                    type_constructor: Optional[Callable[[str], T]] = None,
                    end: str = '\n',
                    question_start: str = ' -> ',
                    confirm_answer: bool = False,
                    default: Union[T, Type[Undefined]] = Undefined,
                    show_default: bool = True,
                    **confirmation_binary_user_question_kwargs) -> T:

    input_message = message + end
    if show_default and default != Undefined:
        input_message += '(' + str(default) + ')'

    input_message += question_start

    if type_constructor is None:
        _type_constructor = cast(Callable[[str], T], output_type)
    else:
        _type_constructor = cast(Callable[[str], T], type_constructor)

    while True:
        raw_str_input = input(input_message)

        # if answer is empty and default is set: ans = default
        if raw_str_input == '' and (default is not Undefined):
            result = cast(T, default)

        else:
            try:
                result = _type_constructor(raw_str_input)
                ensure_type(result, output_type)
            except:
                print(type_error_message)
                continue

        if confirm_answer:
            confirm_message_set = 'message' in confirmation_binary_user_question_kwargs
            if not confirm_message_set:
                confirmation_binary_user_question_kwargs['message'] = f'is {result} correct?'

            if binary_user_question(**confirmation_binary_user_question_kwargs):
                break
            else:
                continue
        else:
            break

    return cast(T, result)


def binary_user_question(message: str,
                         end: str = '\n -> ',
                         error_message: str = 'Invalid answer',
                         true_message: str = 'y',
                         false_message: str = 'n',
                         default: bool = True,
                         exact: bool = False,
                         case_sensitive: bool = False) -> bool:

    y_n_question = f'[{true_message}]/{false_message}' if default is True else f'{true_message}/[{false_message}]'

    while True:
        ans = input(f'{message} {y_n_question}{end}')
        if not case_sensitive:
            ans, true_message, false_message = map(lambda s: s.lower(), (ans, true_message, false_message))
        if exact:
            if ans == true_message:
                return True
            elif ans == false_message:
                return False
        else:
            return ans != false_message if default is True else ans == true_message

        print(error_message)