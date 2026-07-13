"""Exception and warning types. skepsis fails loud: bad input is rejected, never repaired."""


class SkepsisError(Exception):
    """Base class for all skepsis errors."""


class InvalidInputError(SkepsisError):
    """Input is malformed: NaN/inf values, wrong shape, non-numeric data."""


class InsufficientDataError(SkepsisError):
    """Sample is shorter than the diagnostic's documented minimum."""


class MisalignedTrialsError(SkepsisError):
    """returns/trials/params dimensions do not line up."""


class InvalidFrequencyError(SkepsisError):
    """Unrecognized freq value."""


class SkepsisWarning(UserWarning):
    """A statistical assumption is strained; results shown but flagged."""
