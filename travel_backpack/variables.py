
from dataclasses import dataclass
from typing import Any
@dataclass
class VariableReferenceHolder:
    """Holds a reference to a variable

    Useful for passing as argument just like a
    variable pointer or a variable reference in other languages
    """
    value: Any

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

