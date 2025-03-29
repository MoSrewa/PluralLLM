"""
This module defines constants and enumerations used across the application.

Attributes:
    LOG_FILE_LOCATION (str): Path to the file where logs are stored.
"""

from enum import Enum

LOG_FILE_LOCATION = "logs/logs.txt"
"""
Path to the logs file, which is used for recording and analyzing runtime
information, errors, saved models and other messages.
"""

# class ModelConfigKeys(Enum):
#     INPUT_CHANNELS = "input_channels"
#     NUM_CLASSES = "num_classes"


class DatasetKeys(Enum):
    """
    Enumerates the keys for different dataset types available in the project.

    Attributes:
        OQA (str): Represents the key for the OQA dataset.
        GLOBALQA (str): Represents the key for the GlobalQA dataset.
    """
    OQA = "oqa"
    GLOBALQA = "globalqa"


class EmbeddidngModelType(Enum):
    """
    Enumerates types of embedding models used within the application.

    Attributes:
        ALPACA (str): Represents the Alpaca embedding model.
        LLAMA (str): Represents the Llama embedding model.
    """
    ALPACA = "alpaca"
    LLAMA = "llama"
