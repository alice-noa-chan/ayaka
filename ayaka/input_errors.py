"""Input limits shared by offline encoders and the serving boundary."""


class ContextLimitError(ValueError):
    """A complete original input cannot fit without changing its meaning."""
