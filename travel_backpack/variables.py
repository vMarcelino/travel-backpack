
from dataclasses import dataclass
from typing import Any
@dataclass
class VariableReferenceHolder:
    """Holds a reference to a variable

    Useful for passing as argument just like a
    variable pointer or a variable reference in other languages
    """
    value: Any

class NoneClass:
    pass

def check_var_input(message: str, output_type=str, end: str = '\n', question_start:str=' -> ', type_error_message='Invalid type', confirm_answer=True, default=NoneClass, show_default=True, **confirmation_binary_user_question_kwargs):
    """A higher level of the input function.

    Can check for type via output_type parameter, use a default value via
    default parameter, confirm the answer via confirm_answer parameter and
    set custom message parameters via end, type_error_message and kwargs
    that are passed directily to travel_backpack.variables.binary_user_question

    example usage:
        get the port number for a server:
        check_var_input('Server port:',output_type=int, confirm_anwer=False)
        get the same port, but have a default specified
        check_var_input('Server port:',output_type=int, confirm_anwer=False, default=80)
        get the same port, but return None instead of '' in case the user does not enter any information
        check_var_input('Server port:',output_type=int, confirm_anwer=False, default=None)
    
    Arguments:
        message {str} -- The message that will be displayed to the user
    
    Keyword Arguments:
        output_type {Any} -- The type to check against (default: {str})
        end {str} -- The message appended to the end of message before default value (default: {'\n'})
        question_start {str} -- The message appended to the end of message after default value (default: {' -> '})
        type_error_message {str} -- The message to be displayed in case the type does not match (default: {'Invalid type'})
        confirm_answer {bool} -- Whether to confirm the answer after it is inputed (default: {True})
        default {Any} -- What to return in case the user gives an empty response. travel_backpack.variables.NoneClass to disable this feature (default: {NoneClass})
        show_default {bool} -- Whether to show the default value on the question message (default: {True})
        confirmation_binary_user_question_kwargs {kwarg} -- Arguments passed to the confirmation part
    
    Returns:
        Any -- Returns the type specified in output_type
    """
    confirm_message_set = 'message' in confirmation_binary_user_question_kwargs
    input_msg = message+end
    if show_default and default != NoneClass:
        input_msg += '(' + str(default) + ')'
    input_msg += question_start
    while True:
        v = input(message + end)

        # if answer is empty and default is set: ans = default
        if v == '' and default is not NoneClass:
            v = default
        else:
            try:
                v = output_type(v)
            except:
                print(type_error_message)
                continue

        if confirm_answer:
            if not confirm_message_set:
                confirmation_binary_user_question_kwargs['message'] = f'is {v} correct?'

            if binary_user_question(**confirmation_binary_user_question_kwargs):
                return output_type(v)
            else:
                continue
        else:
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

