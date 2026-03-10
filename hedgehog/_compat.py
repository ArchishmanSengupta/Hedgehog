from dataclasses import fields
from typing import get_type_hints, Optional


def get_trainer_config_fields():
    from .trainers import TrainerConfig
    hints = get_type_hints(TrainerConfig)
    result = []
    for f in fields(TrainerConfig):
        ftype = hints.get(f.name, type(f.default))
        result.append((f.name, ftype, f.default))
    return result
