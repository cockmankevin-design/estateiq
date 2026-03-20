"""FastAPI web UI and REST API for the data service layer."""

from .app import create_app
from .api import router

__all__ = ["create_app", "router"]
