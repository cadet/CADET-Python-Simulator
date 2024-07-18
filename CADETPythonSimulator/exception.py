class CADETPythonSimError(Exception):
    """Typical Exception for Error Handling"""
    pass


class NotInitializedError(CADETPythonSimError):
    """Exception raised when a unit operation is not yet initialized."""
    pass