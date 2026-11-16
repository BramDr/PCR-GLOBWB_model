from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import pcraster as pcr


@dataclass(frozen=True)
class CropSoilParams:
    thickness: pcr.Field  # m
    saturated_moisture: pcr.Field  # m3 m-3
    residual_moisture: pcr.Field  # m3 m-3
    effective_saturation_at_field_capacity: pcr.Field  # -
    effective_saturation_at_wilting_point: pcr.Field  # -


@dataclass(frozen=True)
class CropMeteoVars:
    precipitation: pcr.Field  # m day-1
    minimum_temperature: pcr.Field  # degree C
    maximum_temperature: pcr.Field  # degree C
    wind_speed: pcr.Field  # m s-1 at 10m
    shortwave_radiation: pcr.Field  # W m-2
    vapour_pressure: pcr.Field  # Pa


@dataclass(frozen=True)
class CropSoilVars:
    storage: pcr.Field  # m


@dataclass(frozen=True)
class CropOutputs:
    # missing values where no crop is growing
    root_depth: pcr.Field  # m
    crop_coefficient: pcr.Field  # -
    cover_fraction: pcr.Field  # -
    interception_capacity: pcr.Field  # m, per unit cover fraction
    potential_transpiration: pcr.Field  # m day-1
    potential_bare_soil_evaporation: pcr.Field  # m day-1
    water_stress: pcr.Field  # 0-1, actual over potential transpiration
    development_stage: pcr.Field  # 0-2, 0=emergence, 1=flowering, 2=mature


class CropModel(ABC):
    @abstractmethod
    def set_meteo(self, meteo: CropMeteoVars) -> None: ...

    @abstractmethod
    def update(self) -> None: ...


class CropLandCoverModel(ABC):
    # outputs of the last get_outputs; None before the first one
    outputs: CropOutputs | None = None

    @abstractmethod
    def setup(self, model: CropModel) -> None: ...

    @abstractmethod
    def set_soil_params(self, params: Sequence[CropSoilParams]) -> None: ...

    @abstractmethod
    def set_soil_vars(self, vars: Sequence[CropSoilVars]) -> None: ...

    @abstractmethod
    def get_outputs(self) -> CropOutputs: ...
