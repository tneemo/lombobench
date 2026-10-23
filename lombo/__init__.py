"""LomboBench — black-box observability of Ollama's external inference controls.

Phase 1 of LomboAI: transparent proxy + SQLite logging + experimentation Lab.
This package does NOT access model weights/activations; it observes sampling
parameters, context and behavior (metrics, text, thinking).
"""

__version__ = "0.1.0"
