import httpx
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    retry_any,
    retry_all,
    retry_if_not_exception_message,
)


RETRY_ERROR_CLASSES_SIMPLE_CHECK = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    httpx.ReadTimeout,  # happens with OpenAI
)


retry_decorator = retry(
        wait=wait_random_exponential(
            multiplier=1, min=1, max=600
        ),
        stop=stop_after_attempt(30),
        reraise=True,
        retry=retry_any(
            retry_if_exception_type(RETRY_ERROR_CLASSES_SIMPLE_CHECK),
            retry_all(
                retry_if_exception_type(openai.RateLimitError),
                retry_if_not_exception_message(
                    match=r".*You exceeded your current quota, please check your plan and billing details.*"
                ),
            ),
        ),
    )
