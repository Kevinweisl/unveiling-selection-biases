import time
from colorama import Fore, Style

import openai
import google.api_core.exceptions


class AzureRateLimitError(Exception):
    def __init__(self, message="Azure API rate limit exceeded."):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class AzureServerError(Exception):
    def __init__(self, message="Azure API server error."):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    max_retries: int = 10,
    errors: tuple = (
        AzureRateLimitError, AzureServerError,
        openai.RateLimitError, openai.APIError,
        google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.ServiceUnavailable, google.api_core.exceptions.GoogleAPIError,
    ),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1
                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        Fore.RED + f"Maximum number of retries ({max_retries}) exceeded." + Style.RESET_ALL
                    )
                # Increment the delay
                delay *= exponential_base
                # Sleep for the delay
                print(Fore.YELLOW + f"Error encountered. Retry ({num_retries}) after {delay} seconds..." + Style.RESET_ALL)
                time.sleep(delay)
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper
