"""
RepeatLastValues model package.

A simple baseline model that repeats the last observed value for the entire forecast horizon.
"""

from .config import RepeatLastValuesConfig
from .lightning import LitRepeatLastValues
from .model import RepeatLastValues

__all__ = ["RepeatLastValuesConfig", "RepeatLastValues", "LitRepeatLastValues"]
