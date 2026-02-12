import os

def get_package_root():
    """
    Returns the absolute path to the root of the package.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))