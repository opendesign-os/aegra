"""Status enums for Aegra API specification."""

from typing import Literal

# Run status enum
RunStatus = Literal[
    "pending",
    "running",
    "error",
    "success",
    "timeout",
    "interrupted",
]

# Thread status enum
ThreadStatus = Literal[
    "idle",
    "busy",
    "interrupted",
    "error",
]

# Multitask strategy enum
MultitaskStrategy = Literal[
    "reject",
    "rollback",
    "interrupt",
    "enqueue",
]

RunCancellationAction = Literal["cancel", "interrupt"]

Durability = Literal["sync", "async", "exit"]
IfNotExists = Literal["create", "reject"]
