def parallel_for(function, iterable, wait_completion=True, debug=False) -> list:
    if debug:
        for element in iterable:
            function(element)
        return []
    import threading
    threads = []
    for element in iterable:
        t = threading.Thread(target=function, args=(element, ))
        threads.append(t)
        t.start()

    if wait_completion:
        for t in threads:
            t.join()

    return threads


from concurrent.futures import Future, ThreadPoolExecutor, Executor
from typing import Callable, Generic, Optional, TypeVar, Union

T = TypeVar('T')
U = TypeVar('U')


class FutureChain(Generic[T]):
    def __init__(self, future: 'Future[T]', executor: Optional[Executor] = None) -> None:
        self._inner_future = future
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=8)
        self._executor = executor

    def result(self):
        return self._inner_future.result()

    def set_result(self, result: T):
        self._inner_future.set_result(result)

    def add_done_callback(self, cb: Callable[['Future[T]'], None]):
        self._inner_future.add_done_callback(cb)

    def then(self, function: Callable[[T], Union[U]]) -> 'FutureChain[U]':
        def callback():
            result = self.result()
            result = function(result)
            return result

        new_future = self._executor.submit(callback)
        return FutureChain(future=new_future, executor=self._executor)

    def __repr__(self) -> str:
        return f'<{type(self).__name__} {self._inner_future._state} {self._inner_future._result}>'
