#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Data formats exceptions."""


class ParseError(Exception):
    """Base class for bio file parsing errors."""


class AF2MmCIFParseError(ParseError):
    pass


class TemplateError(Exception):
    """Base class for template extraction exceptions."""


class CaDistanceError(TemplateError):
    """An error indicating that a CA atom distance exceeds a threshold."""


class MultipleChainsError(TemplateError):
    """An error indicating that multiple chains were found for a given ID."""
