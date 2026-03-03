"""Qwen 3.5 Scope / QwenScope — SAE-based behavioral decomposition."""

import logging

# Silence noisy HTTP request logging from httpx (used by huggingface_hub).
# Without this, every model/dataset download floods the log with hundreds
# of "HTTP Request: GET ... 200 OK" lines, drowning out actual pipeline output.
logging.getLogger("httpx").setLevel(logging.WARNING)
