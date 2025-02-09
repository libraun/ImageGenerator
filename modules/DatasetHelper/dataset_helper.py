import threading
import itertools
import math

from typing import Iterable
from collections.abc import Callable

def run_threads(dataset: Iterable[any],
                coroutine: Callable[[any], None], # type: ignore
                num_threads: int=2 ) -> None:
    
    if num_threads <= 1 or not dataset:

        raise Exception("""ERROR: Invalid number of threads or invalid dataset. 
                        Ensure you are using more than one thread and that the dataset is a not None.""")

    num_batches = int( math.floor(len(dataset) / num_threads) )
    batches = itertools.batched(dataset, n=num_batches)

    threads = list()

    for batch in batches:
        # Spawn thread for coroutine, assigning current data batch as arg
        thread = threading.Thread(
            target=coroutine, 
            args=[batch])
        threads.append(thread)

    # Call into each thread
    for thread in threads: 
        thread.start()

    # Join each thread
    for thread in threads: 
        thread.join() 