"""
RepeatLastValues model package.

A simple baseline model that repeats the last observed value for the entire forecast horizon.
"""

from .config import RepeatLastValuesConfig
from .model import RepeatLastValues
from .lightning import LitRepeatLastValues

__all__ = ["RepeatLastValuesConfig", "RepeatLastValues", "LitRepeatLastValues"]
