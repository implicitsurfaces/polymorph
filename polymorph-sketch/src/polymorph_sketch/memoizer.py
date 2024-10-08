from functools import wraps
from typing import Any, Callable, Dict, List, TypeVar

T = TypeVar("T")


class Memoizer:
    def __init__(self) -> None:
        self.reset_functions: List[Callable[..., Any]] = []

    def memoize(self) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            cache: Dict[tuple, Any] = {}

            @wraps(func)
            def wrapper(*args: Any) -> T:
                key = tuple(args)
                if key not in cache:
                    cache[key] = func(*args)
                return cache[key]

            def reset_cache() -> None:
                nonlocal cache
                cache.clear()

            self.reset_functions.append(reset_cache)
            return wrapper

        return decorator

    def reset_all_caches(self) -> None:
        for func in self.reset_functions:
            func()
