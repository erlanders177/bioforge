"""Puente: ``bioforge.ai.viability`` -> ``bioforge.evolution.ai.viability``."""

from bioforge.evolution.ai import viability as _target
from bioforge.evolution.ai.viability import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_target, name)


def __dir__():
    return dir(_target)
