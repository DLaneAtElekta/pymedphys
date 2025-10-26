"""MCP Server for Mosaiq Database Failure Mode Testing.

This MCP server provides tools to modify Mosaiq database entries to simulate
various failure modes that could result from third-party writes or data corruption.

This is intended for defensive testing purposes only - to validate error handling
and data validation logic in PyMedPhys.
"""

from .server import main

__all__ = ["main"]
