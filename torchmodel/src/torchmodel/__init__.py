from . import archs
from . import callbacks
from . import datasets
from . import torchmodel
import warnings

__all__ = ['archs', 'callbacks', 'torchmodel', 'datasets']

warnings.filterwarnings(action='ignore', category=UserWarning)

def hello() -> str:
    return "Hello from torchmodel!"
