"""
A few common exceptions for Takahe to use
"""

class TakaheFatalError(Exception):
    pass

class TakaheTypeError(TakaheFatalError):
    pass

"""
And now, warnings:
"""

class TakaheWarning(Warning):
    pass

class TakaheUserWarning(UserWarning, TakaheWarning):
    pass

class TakaheDeprecationWarning(DeprecationWarning, TakaheWarning):
    pass
