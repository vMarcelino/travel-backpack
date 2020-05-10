import inspect
import functools


def decorate_all_methods(decorator, ignore=['__repr__', '__str__']):
    def decorate(cls):
        is_static = lambda attr: isinstance(inspect.getattr_static(cls, attr), (classmethod, staticmethod))
        is_callable = lambda attr: callable(getattr(cls, attr))
        not_to_ignore = lambda attr: attr not in ignore
        filter_func = lambda attr: not_to_ignore(attr) and is_callable(attr) and not is_static(attr)
        for attr in filter(filter_func, cls.__dict__):  # there's propably a better way to do this
            # print('setting method', attr)
            setattr(cls, attr, decorator(getattr(cls, attr)))

        # print(cls, type(cls), 'callable:', callable(cls))
        return cls

    return decorate


def decorate_all_classes_in_module(module, decorator):
    objects = [getattr(module, name) for name in dir(module)]
    # filter_func = lambda obj: inspect.isclass(obj)
    for obj in filter(inspect.isclass, objects):
        decorator(obj)

    return module


def decorate_all_functions_in_module(module, decorator):
    name_obj_pairs = [(name, getattr(module, name)) for name in dir(module)]
    filter_func = lambda pair: callable(pair[1]) and not inspect.isclass(pair[1])
    for name, obj in filter(filter_func, name_obj_pairs):
        setattr(module, name, decorator(obj))

    return module


def decorate_all_modules_in_module(module, decorator, topmost_path=None, only_local_code=True):
    name_obj_pairs = [(name, getattr(module, name)) for name in dir(module)]
    filter_func = lambda pair: inspect.ismodule(pair[1])

    if only_local_code:
        import os
        if topmost_path is None:
            topmost_dir = os.path.dirname(os.path.realpath(module))
        else:
            topmost_dir = os.path.dirname(os.path.realpath(topmost_path))

    for name, obj in filter(filter_func, name_obj_pairs):
        if only_local_code:
            try:
                module_path = os.path.realpath(inspect.getabsfile(obj))
                if topmost_dir in module_path:
                    # print('decorating module', name)
                    setattr(module, name, decorator(obj))
                else:
                    # print('ignoring out-of-scope module', name)
                    pass
            except NotImplementedError:
                raise
            except Exception as ex:
                print('module is absolute:', obj, ex)
        else:
            setattr(module, name, decorator(obj))

    return module


def pure_property(func):
    """A pure property assumes that the result of
    a property will always be the same given a class
    instance. Doing that, the pure property caches
    the result on the first run and returns it in
    the consecutives 

    Arguments:
        func {function} -- The function to be decorated. must have only self as argument

    Returns:
        function descriptor -- the function wrapped in this decorator and the property decorator
    """

    specific_name = f'__pure_{id(func)}'

    @functools.wraps(func)
    def pure_w(self):
        if not hasattr(self, specific_name):
            # print('returning calculated value')
            setattr(self, specific_name, func(self))
        # else:
        #     print('returning cached value')

        return getattr(self, specific_name)

    return property(pure_w)