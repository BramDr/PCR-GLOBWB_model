from __future__ import annotations

import logging
import math
import pathlib as pl
import re
import tempfile
from collections.abc import Sequence

import numpy as np
import pcraster as pcr
import pymt_wofost

from pcrglobwb.crops import (
    CropLandCoverModel,
    CropMeteoVars,
    CropModel,
    CropOutputs,
    CropSoilParams,
    CropSoilVars,
)

logger = logging.getLogger(__name__)

M_TO_CM = 1e2
CM_TO_M = 1e-2
PA_TO_HPA = 1e-2
SECONDS_PER_DAY = 86400.0  # W m-2 to J m-2 day-1
# PCR-GLOBWB wind speed is at 10 m, WOFOST's Penman equations use wind speed at 2 m:
# logarithmic wind profile of Allen et al. (1998), FAO-56 eq. 47
WIND_10M_TO_2M = 4.87 / math.log(67.8 * 10.0 - 5.42)
# Land cover parameters from the leaf area index (Sutanudjaja et al., 2011, A1, A25-26)
CANOPY_EXTINCTION_COEFFICIENT = 0.5  # cover fraction = 1 - exp(-k LAI)
INTERCEPTION_CAPACITY_PER_LAI = 0.0002  # m
CROP_KC_EXTINCTION_COEFFICIENT = 0.7
BARE_SOIL_CROP_KC = 0.2
FULL_COVER_CROP_KC = 1.2


class WofostAdapter(CropModel):

    def __init__(self, config_file: str):
        self.model = pymt_wofost.WOFOST()
        self.config_file = pl.Path(config_file)

        component_name = self.model.get_component_name()
        print(
            f"Initializing {component_name} with configuration file {self.config_file}"
        )

        if not self.config_file.is_file():
            raise FileNotFoundError(
                f"WOFOST configuration file does not exist: {config_file}"
            )
        config = self.config_file.read_text()
        # Make sure BMI_COUPLING is set to COUPLED
        config = re.sub(r"BMI_COUPLING\s.*", "BMI_COUPLING\tCOUPLED", config)

        with tempfile.NamedTemporaryFile("w", suffix=".ini") as temp_file:
            temp_file.write(config)
            temp_file.flush()
            self.model.initialize(temp_file.name)

    def set_meteo(self, meteo: CropMeteoVars):
        # name: PCR-GLOBWB variable converted to WOFOST units
        forcing = {
            "in_precipitation": meteo.precipitation * M_TO_CM,  # cm day-1
            "in_minimum_temperature": meteo.minimum_temperature,  # degC
            "in_maximum_temperature": meteo.maximum_temperature,  # degC
            "in_wind_speed": meteo.wind_speed * WIND_10M_TO_2M,  # m s-1 at 2m
            "in_shortwave_radiation": meteo.shortwave_radiation
            * SECONDS_PER_DAY,  # J m-2 day-1
            "in_vapor_pressure": meteo.vapour_pressure * PA_TO_HPA,  # hPa
        }

        for name, field in forcing.items():
            values = pcr.pcr2numpy(field, np.nan)
            values = np.flipud(values)
            self.model.set_value(name, np.ascontiguousarray(values).ravel())

    def update(self):
        self.model.update()


class WofostLandCoverAdapter(CropLandCoverModel):

    def __init__(self, name: str):
        self.name = name
        self.model = None

        clone = pcr.clone()
        self.shape = (clone.nrRows(), clone.nrCols())

        self.depths: list[pcr.Field] = []
        self.saturated_moistures: list[pcr.Field] = []
        self.residual_moistures: list[pcr.Field] = []
        self.field_capacities: list[pcr.Field] = []
        self.wilting_points: list[pcr.Field] = []

        # WOFOST outputs of the last get_outputs, in PCR-GLOBWB units
        self.outputs: CropOutputs | None = None

    def setup(self, model: WofostAdapter):
        self.model = model.model

    def set_soil_params(self, params: Sequence[CropSoilParams]):
        # Effective saturation (-) to volumetric moisture (m3 m-3)
        def volumetric(p: CropSoilParams, saturation: pcr.Field) -> pcr.Field:
            return p.residual_moisture + saturation * (
                p.saturated_moisture - p.residual_moisture
            )

        # Store the layer properties; set_soil_vars averages them over the root zone
        self.depths = [p.thickness for p in params]
        self.saturated_moistures = [p.saturated_moisture for p in params]
        self.residual_moistures = [p.residual_moisture for p in params]
        self.field_capacities = [
            volumetric(p, p.effective_saturation_at_field_capacity) for p in params
        ]
        self.wilting_points = [
            volumetric(p, p.effective_saturation_at_wilting_point) for p in params
        ]

    def set_soil_vars(self, vars: Sequence[CropSoilVars]):
        assert self.model is not None

        # Root depth of the previous day; without roots, use the top layer
        root_depth = self.depths[0]
        if self.outputs is not None:
            root_depth = pcr.max(pcr.cover(self.outputs.root_depth, 0.0), root_depth)

        # Rooted thickness (m) per layer
        rooted = []
        layer_top = 0.0
        for depth in self.depths:
            rooted.append(pcr.max(0.0, pcr.min(depth, root_depth - layer_top)))
            layer_top = layer_top + depth

        # name: PCR-GLOBWB layer values, averaged over the root zone weighted by
        # rooted thickness; m3 m-3 equals WOFOST's cm cm-3, so no unit conversion
        soil = {
            f"{self.name}_in_soil_saturated": self.saturated_moistures,
            f"{self.name}_in_soil_field_capacity": self.field_capacities,
            f"{self.name}_in_soil_wilting_point": self.wilting_points,
            # storage above residual (m) to volumetric moisture (m3 m-3)
            f"{self.name}_in_soil_saturation": [
                r + v.storage / d
                for v, d, r in zip(
                    vars, self.depths, self.residual_moistures, strict=True
                )
            ],
        }

        depth = sum(rooted)
        for name, layers in soil.items():
            field = sum(v * r for v, r in zip(layers, rooted)) / depth
            values = pcr.pcr2numpy(field, np.nan)
            values = np.flipud(values)
            self.model.set_value(name, np.ascontiguousarray(values).ravel())

    def get_outputs(self) -> CropOutputs:
        assert self.model is not None

        names = [
            "out_root_depth",
            "out_leaf_area_index",
            "out_max_transpiration",
            "out_max_soil_evaporation",
            "out_water_stress",
            "out_development_stage",
        ]

        # WOFOST only writes the cells where it grows the crop; the rest stays NaN
        outputs = {}
        for name in names:
            values = np.full(self.shape, np.nan, dtype=np.float32)
            self.model.get_value(f"{self.name}_{name}", values)
            values = np.flipud(values)
            outputs[name] = pcr.numpy2pcr(
                pcr.Scalar, np.ascontiguousarray(values), np.nan
            )

        # WOFOST variable converted to PCR-GLOBWB units
        lai = outputs["out_leaf_area_index"]  # m2 m-2
        self.outputs = CropOutputs(
            root_depth=outputs["out_root_depth"] * CM_TO_M,  # m
            crop_coefficient=BARE_SOIL_CROP_KC
            + (FULL_COVER_CROP_KC - BARE_SOIL_CROP_KC)
            * (1.0 - pcr.exp(-CROP_KC_EXTINCTION_COEFFICIENT * lai)),  # -
            cover_fraction=1.0 - pcr.exp(-CANOPY_EXTINCTION_COEFFICIENT * lai),  # -
            interception_capacity=INTERCEPTION_CAPACITY_PER_LAI * lai,  # m
            potential_transpiration=outputs["out_max_transpiration"]
            * CM_TO_M,  # m day-1
            potential_bare_soil_evaporation=outputs["out_max_soil_evaporation"]
            * CM_TO_M,  # m day-1
            water_stress=outputs["out_water_stress"],  # -
            development_stage=outputs["out_development_stage"],  # -
        )
        return self.outputs
