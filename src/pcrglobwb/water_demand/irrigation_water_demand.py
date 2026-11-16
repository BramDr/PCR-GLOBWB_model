import logging
import os

import pcraster as pcr

from pcrglobwb.common import virtualOS as vos

logger = logging.getLogger(__name__)


class IrrigationWaterDemand(object):

    def __init__(
        self,
        iniItems,
        nameOfSectionInIniFileThatIsRellevantForThisIrrLC,
        landmask,
        landCoverObject,
    ):
        object.__init__(self)

        self.iniItems = iniItems

        self.cloneMap = iniItems.cloneMap
        self.tmpDir = iniItems.tmpDir
        self.inputDir = iniItems.globalOptions["inputDir"]
        self.landmask = landmask

        self.iniItemsIrrLC = iniItems.__getattribute__(
            nameOfSectionInIniFileThatIsRellevantForThisIrrLC
        )

        self.name = self.iniItemsIrrLC["name"]

        # 'static' parameters (mainly soil and topography)
        self.parameters = landCoverObject.parameters
        self.irrigationType = landCoverObject.irrigationType

        # inundation height allowed within this land cover
        self.minTopWaterLayer = landCoverObject.minTopWaterLayer

        self.cropDeplFactor = vos.readPCRmapClone(
            self.iniItemsIrrLC["cropDeplFactor"],
            self.cloneMap,
            self.tmpDir,
            self.inputDir,
        )

        self.numberOfLayers = landCoverObject.numberOfLayers

        # irrigation efficiency input (value or file name)
        self.ini_items_for_irrigation_efficiency = None
        # by default, use the one in waterDemandOptions
        if "irrigationEfficiency" in self.iniItems.waterDemandOptions.keys():
            self.ini_items_for_irrigation_efficiency = self.iniItems.waterDemandOptions[
                "irrigationEfficiency"
            ]
        # a specific one in the landCoverOptions takes precedence
        if "irrigationEfficiency" in self.iniItemsIrrLC.keys():
            self.ini_items_for_irrigation_efficiency = self.iniItemsIrrLCOptions[
                "irrigationEfficiency"
            ]
        # otherwise, use 1.0
        if self.ini_items_for_irrigation_efficiency is None:
            logger.info("'irrigationEfficiency' is not defined, we set this to 1.0")
            self.ini_items_for_irrigation_efficiency = "1.0"

        # needed in updateLC
        self.includeIrrigation = True

    def get_irrigation_efficiency(self, currTimeStep):
        # irrigation efficiency map (%); TODO: use a time series of efficiency (historical technological development)
        # PCRaster map
        if self.ini_items_for_irrigation_efficiency.endswith(".map"):
            self.irrigationEfficiency = vos.readPCRmapClone(
                self.ini_items_for_irrigation_efficiency,
                self.cloneMap,
                self.tmpDir,
                self.inputDir,
            )
        # netCDF file
        elif "nc" in os.path.splitext(self.ini_items_for_irrigation_efficiency)[1]:
            try:
                # netCDF file with a time dimension
                ncFileIn = vos.getFullPath(
                    self.ini_items_for_irrigation_efficiency, self.inputDir
                )
                self.irrigationEfficiency = vos.netcdf2PCRobjClone(
                    ncFileIn,
                    "automatic",
                    currTimeStep,
                    useDoy="yearly",
                    cloneMapFileName=self.cloneMap,
                )

            except Exception:
                # netCDF file without a time dimension
                msg = (
                    "The file "
                    + (ncFileIn)
                    + " has no time dimension. Constant values will be used."
                )
                logger.warning(msg)
                self.irrigationEfficiency = vos.readPCRmapClone(
                    self.ini_items_for_irrigation_efficiency,
                    self.cloneMap,
                    self.tmpDir,
                    self.inputDir,
                )
            # TODO: remove try/except
        else:
            # constant value
            self.irrigationEfficiency = float(self.ini_items_for_irrigation_efficiency)

        extrapolate = True
        if (
            "noParameterExtrapolation" in self.iniItems.landSurfaceOptions.keys()
            and self.iniItems.landSurfaceOptions["noParameterExtrapolation"] == "True"
        ):
            extrapolate = False

        if extrapolate:
            # extrapolate the efficiency map; TODO: better extrapolation (considering cell size etc.)
            window_size = 1.25 * pcr.clone().cellSize()
            window_size = min(
                window_size,
                min(pcr.clone().nrRows(), pcr.clone().nrCols())
                * pcr.clone().cellSize(),
            )
            try:
                self.irrigationEfficiency = pcr.cover(
                    self.irrigationEfficiency,
                    pcr.windowaverage(self.irrigationEfficiency, window_size),
                )
                self.irrigationEfficiency = pcr.cover(
                    self.irrigationEfficiency,
                    pcr.windowaverage(self.irrigationEfficiency, window_size),
                )
                self.irrigationEfficiency = pcr.cover(
                    self.irrigationEfficiency,
                    pcr.windowaverage(self.irrigationEfficiency, window_size),
                )
                self.irrigationEfficiency = pcr.cover(
                    self.irrigationEfficiency,
                    pcr.windowaverage(self.irrigationEfficiency, window_size),
                )
                self.irrigationEfficiency = pcr.cover(
                    self.irrigationEfficiency,
                    pcr.windowaverage(self.irrigationEfficiency, window_size),
                )
                self.irrigationEfficiency = pcr.cover(
                    self.irrigationEfficiency,
                    pcr.windowaverage(self.irrigationEfficiency, 0.75),
                )
                self.irrigationEfficiency = pcr.cover(
                    self.irrigationEfficiency,
                    pcr.windowaverage(self.irrigationEfficiency, 1.00),
                )
                self.irrigationEfficiency = pcr.cover(
                    self.irrigationEfficiency,
                    pcr.windowaverage(self.irrigationEfficiency, 1.50),
                )
            except Exception:
                pass

        self.irrigationEfficiency = pcr.cover(self.irrigationEfficiency, 1.0)
        self.irrigationEfficiency = pcr.max(0.1, self.irrigationEfficiency)
        self.irrigationEfficiency = pcr.ifthen(self.landmask, self.irrigationEfficiency)

    def calculateTotAvlWaterCapacityInRootZone(self):
        # total water capacity in the root zone (upper soil layers); depends on the land cover type

        if self.numberOfLayers == 2:
            # Edwin uses the soil thicknesses thickUpp and thickLow instead of storCapUpp and storCapLow
            totAvlWater = (
                pcr.max(
                    0.0,
                    self.parameters.effSatAtFieldCapUpp
                    - self.parameters.effSatAtWiltPointUpp,
                )
            ) * (
                self.parameters.satVolMoistContUpp - self.parameters.resVolMoistContUpp
            ) * pcr.min(
                self.parameters.thickUpp, self.rootDepth
            ) + (
                pcr.max(
                    0.0,
                    self.parameters.effSatAtFieldCapLow
                    - self.parameters.effSatAtWiltPointLow,
                )
            ) * (
                self.parameters.satVolMoistContLow - self.parameters.resVolMoistContLow
            ) * pcr.min(
                self.parameters.thickLow,
                pcr.max(self.rootDepth - self.parameters.thickUpp, 0.0),
            )
            # (supported by Rens)
            totAvlWater = pcr.min(
                totAvlWater,
                self.parameters.storCapUpp + self.parameters.storCapLow,
            )

        if self.numberOfLayers == 3:
            totAvlWater = (
                (
                    pcr.max(
                        0.0,
                        self.parameters.effSatAtFieldCapUpp000005
                        - self.parameters.effSatAtWiltPointUpp000005,
                    )
                )
                * (
                    self.parameters.satVolMoistContUpp000005
                    - self.parameters.resVolMoistContUpp000005
                )
                * pcr.min(self.parameters.thickUpp000005, self.rootDepth)
                + (
                    pcr.max(
                        0.0,
                        self.parameters.effSatAtFieldCapUpp005030
                        - self.parameters.effSatAtWiltPointUpp005030,
                    )
                )
                * (
                    self.parameters.satVolMoistContUpp005030
                    - self.parameters.resVolMoistContUpp005030
                )
                * pcr.min(
                    self.parameters.thickUpp005030,
                    pcr.max(0.0, self.rootDepth - self.parameters.thickUpp000005),
                )
                + (
                    pcr.max(
                        0.0,
                        self.parameters.effSatAtFieldCapLow030150
                        - self.parameters.effSatAtWiltPointLow030150,
                    )
                )
                * (
                    self.parameters.satVolMoistContLow030150
                    - self.parameters.resVolMoistContLow030150
                )
                * pcr.min(
                    self.parameters.thickLow030150,
                    pcr.max(
                        self.rootDepth
                        - self.parameters.thickUpp000005
                        - self.parameters.thickUpp005030,
                        0.0,
                    ),
                )
            )
            totAvlWater = pcr.min(
                totAvlWater,
                self.parameters.storCapUpp000005
                + self.parameters.storCapUpp005030
                + self.parameters.storCapLow030150,
            )

        return totAvlWater

    def get_readily_available_water_within_the_root_zone(self):

        if self.numberOfLayers == 2:
            effSatUpp = vos.getValDivZero(self.storUpp, self.parameters.storCapUpp)
            effSatLow = vos.getValDivZero(self.storLow, self.parameters.storCapLow)
            effSatUpp = pcr.min(1.0, effSatUpp)
            effSatLow = pcr.min(1.0, effSatLow)

            # readily available water in the root zone (upper soil layers)
            readAvlWater = (
                pcr.max(0.0, effSatUpp - self.parameters.effSatAtWiltPointUpp)
            ) * (
                self.parameters.satVolMoistContUpp - self.parameters.resVolMoistContUpp
            ) * pcr.min(
                self.parameters.thickUpp, self.rootDepth
            ) + (
                pcr.max(0.0, effSatLow - self.parameters.effSatAtWiltPointLow)
            ) * (
                self.parameters.satVolMoistContLow - self.parameters.resVolMoistContLow
            ) * pcr.min(
                self.parameters.thickLow,
                pcr.max(self.rootDepth - self.parameters.thickUpp, 0.0),
            )

        if self.numberOfLayers == 3:
            # effective degree of saturation (-)
            effSatUpp000005 = vos.getValDivZero(
                self.storUpp000005, self.parameters.storCapUpp000005
            )
            effSatUpp005030 = vos.getValDivZero(
                self.storUpp005030, self.parameters.storCapUpp005030
            )
            effSatLow030150 = vos.getValDivZero(
                self.storLow030150, self.parameters.storCapLow030150
            )
            effSatUpp000005 = pcr.min(1.0, effSatUpp000005)
            effSatUpp005030 = pcr.min(1.0, effSatUpp005030)
            effSatLow030150 = pcr.min(1.0, effSatLow030150)

            # readily available water in the root zone (upper soil layers)
            readAvlWater = (
                (
                    pcr.max(
                        0.0,
                        effSatUpp000005 - self.parameters.effSatAtWiltPointUpp000005,
                    )
                )
                * (
                    self.parameters.satVolMoistContUpp000005
                    - self.parameters.resVolMoistContUpp000005
                )
                * pcr.min(self.parameters.thickUpp000005, self.rootDepth)
                + (
                    pcr.max(
                        0.0,
                        effSatUpp005030 - self.parameters.effSatAtWiltPointUpp005030,
                    )
                )
                * (
                    self.parameters.satVolMoistContUpp005030
                    - self.parameters.resVolMoistContUpp005030
                )
                * pcr.min(
                    self.parameters.thickUpp005030,
                    pcr.max(0.0, self.rootDepth - self.parameters.thickUpp000005),
                )
                + (
                    pcr.max(
                        0.0,
                        effSatLow030150 - self.parameters.effSatAtWiltPointLow030150,
                    )
                )
                * (
                    self.parameters.satVolMoistContLow030150
                    - self.parameters.resVolMoistContLow030150
                )
                * pcr.min(
                    self.parameters.thickLow030150,
                    pcr.max(
                        self.rootDepth
                        - self.parameters.thickUpp000005
                        - self.parameters.thickUpp005030,
                        0.0,
                    ),
                )
            )

        return readAvlWater

    def update(self, meteo, landSurface, groundwater, routing, currTimeStep):

        # variables from landSurface.landCoverObj
        self.cropKC = landSurface.landCoverObj[self.name].cropKC
        self.minCropKC = landSurface.landCoverObj[self.name].minCropKC
        self.topWaterLayer = landSurface.landCoverObj[self.name].topWaterLayer
        self.rootDepth = landSurface.landCoverObj[self.name].rootDepth
        self.wofost = landSurface.landCoverObj[self.name].wofost

        # soil states from landSurface.landCoverObj
        if self.numberOfLayers == 2:
            self.storUpp = landSurface.landCoverObj[self.name].storUpp
            self.storLow = landSurface.landCoverObj[self.name].storLow
            self.soilWaterStorage = self.storUpp + self.storLow

        if self.numberOfLayers == 3:
            self.storUpp000005 = landSurface.landCoverObj[self.name].storUpp000005
            self.storUpp005030 = landSurface.landCoverObj[self.name].storUpp005030
            self.storLow030150 = landSurface.landCoverObj[self.name].storLow030150
            self.soilWaterStorage = (
                self.storUpp000005 + self.storUpp005030 + self.storLow030150
            )

        self.totAvlWater = self.calculateTotAvlWaterCapacityInRootZone()
        self.readAvlWater = self.get_readily_available_water_within_the_root_zone()

        # more variables from landSurface.landCoverObj
        self.netLqWaterToSoil = landSurface.landCoverObj[self.name].netLqWaterToSoil
        self.fracVegCover = landSurface.landCoverObj[self.name].fracVegCover
        self.totalPotET = landSurface.landCoverObj[self.name].totalPotET

        # irrigation efficiency (yearly; sets self.irrigationEfficiency)
        if currTimeStep.doy == 1 or currTimeStep.timeStepPCR == 1:
            self.get_irrigation_efficiency(currTimeStep)

        # irrigation water demand (m/day) for paddy and non-paddy fields;
        self.irrGrossDemand = pcr.scalar(0.0)
        if self.irrigationType == "irrPaddy":

            irrTrigger = self.cropKC > 0.75
            if self.wofost is not None:
                outputs = self.wofost.outputs
                assert outputs is not None
                developmentStage = pcr.cover(outputs.development_stage, 2)
                irrTrigger = irrTrigger | (developmentStage < 1.0)

            self.irrGrossDemand = pcr.ifthenelse(
                irrTrigger,
                pcr.max(0.0, self.minTopWaterLayer - (self.topWaterLayer)),
                0.0,
            )

        if self.irrigationType == "irrNonPaddy" and self.includeIrrigation:

            if self.wofost is None:
                # original formula from Allen et al. (1998); see http://www.fao.org/docrep/x0490e/x0490e0e.html
                adjDeplFactor = pcr.max(
                    0.1,
                    pcr.min(
                        0.8,
                        (self.cropDeplFactor + 0.04 * (5.0 - self.totalPotET * 1000.0)),
                    ),
                )
                irrTrigger = (
                    self.readAvlWater < (1.0 - adjDeplFactor) * self.totAvlWater
                )
            else:
                outputs = self.wofost.outputs
                assert outputs is not None
                stressFactor = pcr.cover(outputs.water_stress, 1.0)
                irrTrigger = (self.readAvlWater < self.totAvlWater) & (
                    stressFactor < 1.0
                )

            self.irrGrossDemand = pcr.ifthenelse(
                self.cropKC > self.minCropKC,
                pcr.ifthenelse(
                    irrTrigger,
                    pcr.max(
                        0.0,
                        self.totAvlWater - self.readAvlWater,
                    ),
                    0.0,
                ),
                0.0,
            )

            # limit the demand by the potential evaporation of the coming days, to avoid
            # unrealistically high demands
            max_irrigation_interval = 15.0
            min_irrigation_interval = 7.0
            irrigation_interval = pcr.min(
                max_irrigation_interval,
                pcr.max(
                    min_irrigation_interval,
                    pcr.ifthenelse(
                        self.totalPotET > 0.0,
                        pcr.roundup(
                            (
                                self.irrGrossDemand
                                + pcr.max(self.readAvlWater, self.soilWaterStorage)
                            )
                            / self.totalPotET
                        ),
                        1.0,
                    ),
                ),
            )

            # irrigation demand, limited by the potential evaporation of the coming days
            self.irrGrossDemand = pcr.min(
                pcr.max(
                    0.0,
                    self.totalPotET * irrigation_interval
                    - pcr.max(self.readAvlWater, self.soilWaterStorage),
                ),
                self.irrGrossDemand,
            )

            # assume that smart farmers do not irrigate more than the infiltration capacity
            if self.numberOfLayers == 2:
                self.irrGrossDemand = pcr.min(
                    self.irrGrossDemand, self.parameters.kSatUpp
                )
            if self.numberOfLayers == 3:
                self.irrGrossDemand = pcr.min(
                    self.irrGrossDemand, self.parameters.kSatUpp000005
                )

        # irrigation efficiency, minimum demand to start irrigating and maximum demand
        if self.includeIrrigation:

            # irrigation efficiency; TODO: improve the concept of irrigation efficiency
            self.irrigationEfficiencyUsed = pcr.min(
                1.0, pcr.max(0.10, self.irrigationEfficiency)
            )

            # demand including inefficiency
            self.irrGrossDemand = pcr.cover(
                self.irrGrossDemand / pcr.min(1.0, self.irrigationEfficiencyUsed), 0.0
            )

            # this demand is not limited by the available water
            self.irrGrossDemand = pcr.ifthen(self.landmask, self.irrGrossDemand)

            # reduce irrGrossDemand by netLqWaterToSoil
            self.irrGrossDemand = pcr.max(
                0.0, self.irrGrossDemand - self.netLqWaterToSoil
            )

            # minimum demand to start irrigating (m/day); TODO: set in the ini file
            minimum_demand = 0.005
            if self.irrigationType == "irrPaddy":
                minimum_demand = pcr.min(self.minTopWaterLayer, 0.025)
            self.irrGrossDemand = pcr.ifthenelse(
                self.irrGrossDemand > minimum_demand, self.irrGrossDemand, 0.0
            )

            # maximum demand (m/day); TODO: set in the ini file
            maximum_demand = 0.025
            if self.irrigationType == "irrPaddy":
                maximum_demand = pcr.min(self.minTopWaterLayer, 0.025)
            self.irrGrossDemand = pcr.min(maximum_demand, self.irrGrossDemand)

            # ignore small demands (less than 1 mm)
            self.irrGrossDemand = pcr.rounddown(self.irrGrossDemand * 1000.0) / 1000.0

            # only for areas with fracVegCover > 0; TODO: check whether this is needed
            self.irrGrossDemand = pcr.ifthenelse(
                self.fracVegCover > 0.0, self.irrGrossDemand, 0.0
            )

        # total gross irrigation demand (m) per cover type (not limited by available water)
        self.totalPotentialMaximumIrrGrossDemandPaddy = 0.0
        self.totalPotentialMaximumIrrGrossDemandNonPaddy = 0.0

        if self.irrigationType == "irrPaddy":
            self.totalPotentialMaximumIrrGrossDemandPaddy = self.irrGrossDemand
        if self.irrigationType == "irrNonPaddy":
            self.totalPotentialMaximumIrrGrossDemandNonPaddy = self.irrGrossDemand
