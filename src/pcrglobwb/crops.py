# crops.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pcraster as pcr
from pcrglobwb.meteo import Meteo


@dataclass(frozen=True)
class CropOutputs:
    root_depth: pcr.Field  # m; 0 where no crop
    leaf_area_index: pcr.Field  # m2 m-2
    max_transpiration: pcr.Field  # m day-1
    max_soil_evaporation: pcr.Field  # m day-1
    water_stress: pcr.Field  # 0..1


class CropModel(ABC):
    @abstractmethod
    def update(self, meteo: Meteo) -> None: ...
