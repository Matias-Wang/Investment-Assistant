import asyncio
import functools
import random
import time
from typing import Callable, Any, Type, Tuple, Optional
from pydantic import ValidationError
from app.core.logging import logger
from app.core.config import settings

NON_RETRYABLE_EXCEPTIONS = (
    ValueError,
    TypeError,
    ValidationError,
    KeyError,
)


def is_retryable(exc: Exception) -> bool:
    """判斷例外是否值得重試 (排除驗證錯誤與 4xx 客戶端錯誤)"""
    if isinstance(exc, NON_RETRYABLE_EXCEPTIONS):
        return False
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status_code, int) and 400 <= status_code < 500 and status_code != 429:
        return False
    return True


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator for synchronous functions with retry and exponential backoff with jitter.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if not is_retryable(e):
                        logger.warning(f"[Retry Skipped] Function {func.__name__} raised non-retryable error: {e}")
                        raise

                    if attempt == max_attempts:
                        logger.error(
                            f"[Retry Failed] Function {func.__name__} failed after {max_attempts} attempts. Error: {e}"
                        )
                        raise

                    sleep_time = delay + random.uniform(0, 0.3 * delay)
                    logger.warning(
                        f"[Retry Attempt {attempt}/{max_attempts}] Function {func.__name__} raised {e.__class__.__name__}: {e}. Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                    delay *= backoff_factor
            if last_exception:
                raise last_exception
        return wrapper
    return decorator


def with_async_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator for asynchronous functions with retry and exponential backoff with jitter.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if not is_retryable(e):
                        logger.warning(f"[Async Retry Skipped] Function {func.__name__} raised non-retryable error: {e}")
                        raise

                    if attempt == max_attempts:
                        logger.error(
                            f"[Async Retry Failed] Function {func.__name__} failed after {max_attempts} attempts. Error: {e}"
                        )
                        raise

                    sleep_time = delay + random.uniform(0, 0.3 * delay)
                    logger.warning(
                        f"[Async Retry Attempt {attempt}/{max_attempts}] Function {func.__name__} raised {e.__class__.__name__}: {e}. Retrying in {sleep_time:.2f}s..."
                    )
                    await asyncio.sleep(sleep_time)
                    delay *= backoff_factor
            if last_exception:
                raise last_exception
        return wrapper
    return decorator
