"""
A few common exceptions for Takahe to use
"""


class TakaheFatalError(Exception):
    """A fatal, unrecoverable error raised by Takahe."""
    pass


class TakaheTypeError(TakaheFatalError):
    """A fatal error raised when Takahe receives a malformed type."""
    pass


"""
And now, warnings:
"""


class TakaheWarning(Warning):
    """The base class for all non-fatal warnings raised by Takahe."""
    pass


class TakaheUserWarning(UserWarning, TakaheWarning):
    """A warning raised in response to potentially misguided user input."""
    pass


class TakaheDeprecationWarning(DeprecationWarning, TakaheWarning):
    """A warning raised when using a deprecated Takahe feature."""
    pass
