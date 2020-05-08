from typing import List, Union
time_to_string = lambda x: '%04d/%02d/%02d - %02d:%02d:%02d' % (x.year, x.month, x.day, x.hour, x.minute, x.second)

def time_now_to_string(separators:List[str]=None, order:List[str]=None, lengths:Union[List[int], int]=None) -> str:
    '''
    defaults:

    order = ['y', 'mo', 'd', 'h', 'mi', 's']
    
    separators = ['/', '/', ' - ', ':', ':', '.']
    
    lengths = [4, 2, 2, 2, 2, 2, 4]

    Order can also be changed to show less data. If so, separators must be changed accordingly
    '''
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

def format_date_to_path(date):
    return date.replace('/', '-').replace(':', 'h').replace('_', ' ')
