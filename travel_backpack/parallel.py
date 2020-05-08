def parallel_for(function, iterable, wait_completion=True, debug=False)->list:
    if debug:
        for element in iterable:
            function(element)
        return
    import threading
    threads = []
    for element in iterable:
        t = threading.Thread(target=function, args=(element,))
        threads.append(t)
        t.start()

    if wait_completion:
        for t in threads:
            t.join()

    return threads
