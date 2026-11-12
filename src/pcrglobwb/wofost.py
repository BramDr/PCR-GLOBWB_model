import pathlib as pl
import re
import tempfile

from pcrglobwb.landSurface import LandSurface
from pcrglobwb.landCover import LandCover
from pcrglobwb.parameterSoilAndTopo import SoilAndTopoParameters
from pcrglobwb.meteo import Meteo
import numpy as np
import pymt_wofost
import pcraster as pcr


class WofostAdapter:
    def __init__(self, iniItems, landsurface: LandSurface):
        self.iniItems = iniItems
        self.landsurface = landsurface

        self.options = self.iniItems.landSurfaceOptions
        self.coverModels: list[WofostLandCoverAdapter] = []

        config_file = self.options.get("configuration_file_for_wofost")
        if config_file is None:
            raise ValueError(
                "Configuration file for WOFOST is not specified in the ini file."
            )

        self.model = pymt_wofost.WOFOST()
        component_name = self.model.get_component_name()
        print(f"Initializing {component_name} with configuration file {config_file}")

        config_file = pl.Path(config_file)
        if not config_file.exists():
            raise ValueError(
                f"Configuration file for WOFOST does not exist: {config_file}"
            )

        config = config_file.read_text()
        # Make sure BMI_COUPLING is set to COUPLED
        config = re.sub(r"BMI_COUPLING\s.*", "BMI_COUPLING\tCOUPLED", config)
        with tempfile.TemporaryNamedFile() as temp_file:
            temp_file.write(config.encode())
            temp_file.seek(0)
            self.model.initialize(temp_file.name)

    def setup(self):
        self.coverModels = []
        for coverType in self.landsurface.coverTypes:
            landcover: LandCover = self.landsurface.landCoverObj[coverType]
            coverModel = landcover.wofost
            if coverModel is None:
                continue
            self.coverModels.append(coverModel)

        for coverModel in self.coverModels:
            coverModel.setup(self.model)

    def update(self, meteo: Meteo):
        assert isinstance(meteo.precipitation, pcr.Field)
        assert isinstance(meteo.air_temperature_min, pcr.Field)
        assert isinstance(meteo.air_temperature_max, pcr.Field)
        assert isinstance(meteo.wind_speed_10m, pcr.Field)
        assert isinstance(meteo.shortwave_radiation, pcr.Field)
        assert isinstance(meteo.vapour_pressure, pcr.Field)

        # Aquire meteorological variables
        precipiation = meteo.precipitation
        minimum_temperature = meteo.air_temperature_min
        maximum_temperature = meteo.air_temperature_max
        wind_speed = meteo.wind_speed_10m
        shortwave_radiation = meteo.shortwave_radiation
        vapour_pressure = meteo.vapour_pressure

        # Convert to numpy
        precipiation = pcr.pcr2numpy(precipiation, np.nan)
        minimum_temperature = pcr.pcr2numpy(minimum_temperature, np.nan)
        maximum_temperature = pcr.pcr2numpy(maximum_temperature, np.nan)
        wind_speed = pcr.pcr2numpy(wind_speed, np.nan)
        shortwave_radiation = pcr.pcr2numpy(shortwave_radiation, np.nan)
        vapour_pressure = pcr.pcr2numpy(vapour_pressure, np.nan)

        # Convert units
        precipiation *= 1e2  # m to cm
        shortwave_radiation *= 86400  # W m-2 to J m-2 day-1
        vapour_pressure *= 0.01  # Pa to hPa

        # Set the meteorological variables in the model
        self.model.set_value("in_precipitation", precipiation)
        self.model.set_value("in_minimum_temperature", minimum_temperature)
        self.model.set_value("in_maximum_temperature", maximum_temperature)
        self.model.set_value("in_wind_speed", wind_speed)
        self.model.set_value("in_shortwave_radiation", shortwave_radiation)
        self.model.set_value("in_vapor_pressure", vapour_pressure)

        for coverModel in self.coverModels:
            coverModel.pre_update(self.model)

        self.model.update()

        for coverModel in self.coverModels:
            coverModel.post_update(self.model, meteo)


class WofostLandCoverAdapter:
    def __init__(self, name: str, landcover: LandCover):
        self.name = name
        self.landcover = landcover

        self.params: SoilAndTopoParameters = self.landcover.parameters
        self.shape = (pcr.clone().nrRows, pcr.clone().nrCols)

        self.model: pymt_wofost.WOFOST | None = None

        # Inputs
        self.depths: list[pcr.Field] = []
        self.saturated_moistures: list[pcr.Field] = []
        self.residual_moistures: list[pcr.Field] = []

        # Outputs
        self.root_depth: pcr.Field | None = None
        self.leaf_area_index: pcr.Field | None = None
        self.max_transpiration: pcr.Field | None = None
        self.max_soil_evaporation: pcr.Field | None = None
        self.water_stress: pcr.Field | None = None

    def setup(self, model: pymt_wofost.WOFOST):

        # Aquire layer soil properties
        depths: list[pcr.Field] = []
        saturated_moistures: list[pcr.Field] = []
        aerated_moistures: list[pcr.Field] = []
        field_capacities: list[pcr.Field] = []
        wilting_points: list[pcr.Field] = []
        residual_moistures: list[pcr.Field] = []
        if self.params.numberOfLayers == 2:
            depths = [
                self.params.thickUpp,
                self.params.thickLow,
            ]
            saturated_moistures = [
                self.params.satVolMoistContUpp,
                self.params.satVolMoistContLow,
            ]
            aerated_moistures = [
                self.params.airEntryValueUpp,
                self.params.airEntryValueLow,
            ]
            field_capacities = [
                self.params.effSatAtFieldCapUpp,
                self.params.effSatAtFieldCapLow,
            ]
            wilting_points = [
                self.params.effSatAtWiltPointUpp,
                self.params.effSatAtWiltPointLow,
            ]
            residual_moistures = [
                self.params.resVolMoistContUpp,
                self.params.resVolMoistContLow,
            ]
        elif self.params.numberOfLayers == 3:
            depths = [
                self.params.thickUpp000005,
                self.params.thickUpp005030,
                self.params.thickLow030150,
            ]
            saturated_moistures = [
                self.params.satVolMoistContUpp000005,
                self.params.satVolMoistContUpp005030,
                self.params.satVolMoistContLow030150,
            ]
            aerated_moistures = [
                self.params.airEntryValueUpp000005,
                self.params.airEntryValueUpp005030,
                self.params.airEntryValueLow030150,
            ]
            field_capacities = [
                self.params.effSatAtFieldCapUpp000005,
                self.params.effSatAtFieldCapUpp005030,
                self.params.effSatAtFieldCapLow030150,
            ]
            wilting_points = [
                self.params.effSatAtWiltPointUpp000005,
                self.params.effSatAtWiltPointUpp005030,
                self.params.effSatAtWiltPointLow030150,
            ]
            residual_moistures = [
                self.params.resVolMoistContUpp000005,
                self.params.resVolMoistContUpp005030,
                self.params.resVolMoistContLow030150,
            ]
        else:
            raise ValueError(
                f"Unsupported number of soil layers: {self.params.numberOfLayers}"
            )

        # Convert to numpy
        depths = [pcr.pcr2numpy(m, np.nan) for m in depths]
        saturated_moistures = [pcr.pcr2numpy(m, np.nan) for m in saturated_moistures]
        aerated_moistures = [pcr.pcr2numpy(m, np.nan) for m in aerated_moistures]
        field_capacities = [pcr.pcr2numpy(m, np.nan) for m in field_capacities]
        wilting_points = [pcr.pcr2numpy(m, np.nan) for m in wilting_points]
        residual_moistures = [pcr.pcr2numpy(m, np.nan) for m in residual_moistures]

        # Average soil properties weighted by layer thickness
        depth = sum(depths)
        saturated_moisture = (
            sum(m * d for m, d in zip(saturated_moistures, depths)) / depth
        )
        aerated_moisture = sum(m * d for m, d in zip(aerated_moistures, depths)) / depth
        field_capacity = sum(m * d for m, d in zip(field_capacities, depths)) / depth
        wilting_point = sum(m * d for m, d in zip(wilting_points, depths)) / depth

        # Store inputs required later
        self.depths = depths
        self.saturated_moistures = saturated_moistures
        self.residual_moistures = residual_moistures

        # Set the averaged soil properties in the model
        model.set_value(f"{self.name}_in_soil_saturated", saturated_moisture)
        model.set_value(f"{self.name}_in_soil_aerated", aerated_moisture)
        model.set_value(f"{self.name}_in_soil_field_capacity", field_capacity)
        model.set_value(f"{self.name}_in_soil_wilting_point", wilting_point)

    def pre_update(self, model: pymt_wofost.WOFOST):
        root_depth = model.get_value(f"{self.name}_out_root_depth")

        moistures: list[pcr.Field] = []
        if self.landcover.numberOfSoilLayers == 2:
            moistures.append(self.landcover.storUpp)
            moistures.append(self.landcover.storLow)
        elif self.landcover.numberOfSoilLayers == 3:
            moistures.append(self.landcover.storUpp000005)
            moistures.append(self.landcover.storUpp005030)
            moistures.append(self.landcover.storLow030150)
        else:
            raise ValueError(
                f"Unsupported number of soil layers: {self.landcover.numberOfSoilLayers}"
            )

        # Convert to numpy
        moistures = [pcr.pcr2numpy(m, np.nan) for m in moistures]

        # Aquire soil moisture components within the root zone
        moisture = 0
        saturated_moisture = 0
        residual_moisture = 0
        for i, depth in enumerate(self.depths):
            if root_depth > depth + self.depths[i] and depth > 0:
                continue
            moisture += moistures[i]
            saturated_moisture += self.saturated_moistures[i] * self.depths[i]
            residual_moisture += self.residual_moistures[i] * self.depths[i]

        # Calculate soil saturation
        saturation = (moisture - residual_moisture) / (
            saturated_moisture - residual_moisture
        )
        saturation = np.clip(saturation, 0, 1)

        # Set the soil saturation in the model
        model.set_value(f"{self.name}_in_soil_saturation", saturation)

    def post_update(self, model: pymt_wofost.WOFOST, meteo: Meteo):
        assert isinstance(self.meteo.referencePotET, pcr.Field)

        # Allocate WOFOST variables
        root_depth = np.full(self.shape, np.nan, dtype=np.float32)
        leaf_area_index = np.full(self.shape, np.nan, dtype=np.float32)
        max_transpiration = np.full(self.shape, np.nan, dtype=np.float32)
        max_soil_evaporation = np.full(self.shape, np.nan, dtype=np.float32)
        water_stress = np.full(self.shape, np.nan, dtype=np.float32)

        # Aquire WOFOST variables
        model.get_value(f"{self.name}_out_root_depth", root_depth)
        model.get_value(f"{self.name}_out_leaf_area_index", leaf_area_index)
        model.get_value(f"{self.name}_out_max_transpiration", max_transpiration)
        model.get_value(f"{self.name}_out_max_soil_evaporation", max_soil_evaporation)
        model.get_value(f"{self.name}_out_water_stress", water_stress)

        # Convert units
        root_depth *= 1e-2  # cm to m
        max_transpiration *= 1e-2  # cm to m
        max_soil_evaporation *= 1e-2  # cm to m

        # Convert to PCRaster fields
        self.root_depth = pcr.numpy2pcr(pcr.Scalar, root_depth, np.nan)
        self.leaf_area_index = pcr.numpy2pcr(pcr.Scalar, leaf_area_index, np.nan)
        self.max_transpiration = pcr.numpy2pcr(pcr.Scalar, max_transpiration, np.nan)
        self.max_soil_evaporation = pcr.numpy2pcr(
            pcr.Scalar, max_soil_evaporation, np.nan
        )
        self.water_stress = pcr.numpy2pcr(pcr.Scalar, water_stress, np.nan)
