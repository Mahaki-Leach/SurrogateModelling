# Import Python libraries
import logging
import os

# Import Pyomo libraries
from pyomo.environ import (
    Reference,
    Expression,
    Constraint,
    Param,
    Reals,
    value,
    Var,
    NonNegativeReals,
    units,
)

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    PhysicalParameterBlock,
    StateBlockData,
    StateBlock,
    MaterialBalanceType,
    EnergyBalanceType,
    VaporPhase,
    Component,
)
from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
import idaes.logger as idaeslog

from pyomo.environ import Block, Constraint
from pyomo.core.base.expression import ScalarExpression, Expression, _GeneralExpressionData, ExpressionData
from pyomo.core.base.var import ScalarVar, _GeneralVarData, VarData, IndexedVar, Var

from pyomo.environ import (
    Block,
    check_optimal_termination,
    Constraint,
    exp,
    Expression,
    log,
    Set,
    Param,
    value,
    Var,
    units as pyunits,
    Reference,
)
from pyomo.common.config import ConfigBlock, ConfigDict, ConfigValue, In, Bool
from pyomo.util.calc_var_value import calculate_variable_from_constraint

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    PhysicalParameterBlock,
    StateBlockData,
    StateBlock,
    MaterialFlowBasis,
    ElectrolytePropertySet,
)
from idaes.core.base.components import Component, __all_components__
from idaes.core.base.phases import (
    Phase,
    AqueousPhase,
    LiquidPhase,
    VaporPhase,
    __all_phases__,
)
from idaes.core.util.initialization import (
    fix_state_vars,
    revert_state_vars,
    solve_indexed_blocks,
)
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_activated_constraints,
)
from idaes.core.util.exceptions import (
    BurntToast,
    ConfigurationError,
    PropertyPackageError,
    PropertyNotSupportedError,
    InitializationError,
)
from idaes.core.util.misc import add_object_reference
from idaes.core.solvers import get_solver
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale
from idaes.core.initialization.initializer_base import InitializerBase

from idaes.models.properties.modular_properties.base.generic_reaction import (
    equil_rxn_config,
)
from idaes.models.properties.modular_properties.base.utility import (
    get_method,
    get_phase_method,
    GenericPropertyPackageError,
    StateIndex,
    identify_VL_component_list,
    estimate_Tbub,
    estimate_Tdew,
    estimate_Pbub,
    estimate_Pdew,
)
from idaes.models.properties.modular_properties.phase_equil.bubble_dew import (
    LogBubbleDew,
)
from idaes.models.properties.modular_properties.phase_equil.henry import HenryType

from idaes.models.properties.modular_properties.base.generic_property import GenericParameterData


# Manager class for surrogate state blocks
class _SurrogateStateBlock(StateBlock):

    def initialize(self, *args, **kwargs):
        return super().initialize(*args, **kwargs)


@declare_process_block_class("SurrogateStateBlock", block_class=_SurrogateStateBlock)
class SurrogateStateBlockData(StateBlockData):
    """
    Surrogate state block class for surrogate property packages.
    This class is used to define the state variables and constraints
    for the surrogate model.
    """

    def build(self):
        """
        Build method for the surrogate state block.
        """
        # Calling parent build method
        super(SurrogateStateBlockData, self).build()

        # Build state variables
        self.build_state_vars()
    

    def build_state_vars(self):
        """
        Build the state variables for the surrogate state block.
        This method is called during the block construction.
        """

        self.flow_mol = Var(
            domain=NonNegativeReals,
            initialize=1.0,
            units=units.mol / units.s,
            doc="Total molar flowrate [mol/s]",
        )

        self.pressure = Var(
            domain=NonNegativeReals,
            initialize=95000,
            bounds=(10000, 900000),
            units=units.Pa,
            doc="State pressure [Pa]",
        )

        self.temperature = Var(
            domain=NonNegativeReals,
            initialize=350,
            bounds=(193.15, 273.15+250),
            units=units.K,
            doc="Dry bulb temperature [K]",
        )

        self.mole_frac_comp = Var(
            self.params.component_list,
            initialize = 1/len(self.params.component_list),
            domain=Reals,
            bounds = (0,1),
            units=units.dimensionless
        )

        self.enth_mol = Var(
            domain=Reals,
            initialize=300,
            units = units.J / units.mol,
            doc = "Enthalpy [J/mol]"
        )

        self.entr_mol = Var(
            domain=Reals,
            initialize=40,
            units = units.J / units.mol / units.K,
            doc = "Entropy [J/mol/K]"
        )

        self.vol_mol = Var(
            domain = NonNegativeReals,
            initialize=40,
            units = units.m**3 / units.mol
        )

        inputs = [self.temperature, self.pressure, self.mole_frac_comp["benzene"], self.mole_frac_comp["toluene"]]
        outputs = [self.enth_mol, self.entr_mol, self.vol_mol]
        script_dir = os.path.dirname(__file__)

        self.pysmo_surrogate = PysmoSurrogate.load_from_file(
            os.path.join(script_dir,"pysmo_humid_air.json")
        )

        self.surrogate = SurrogateBlock()

        self.surrogate.build_model(
            self.pysmo_surrogate,
            input_vars=inputs,
            output_vars=outputs,
        )
    
    def _vol_mass(self):
        def _vol_mass_rule(b):
            return b.vol_mol / sum(
                b.mole_frac_comp[i] 
                * (1/b.params.mw_comp[i])
                for i in b.params.component_list
                )
        self.vol_mass = Expression(rule=_vol_mass_rule)

    def _enth_mass(self):
        def enth_mass_rule(b):
            return sum(b.enth_mass_comp[i] for i in b.params.component_list)
        self.enth_mass = Expression (rule=enth_mass_rule)

    def _enth_mass_comp(self):
        def _rule_enth_mass_comp(b, i):
            return b.enth_mol_comp[i] / b.params.mw_comp[i]
        self.enth_mass_comp = Expression(
            self.params.component_list,
            rule=_rule_enth_mass_comp,
        )

    def _enth_mol_comp(self): ### check this
        def _rule_enth_mol_comp(b, i):
            return b.enth_mol * b.mole_frac_comp[i]
        self.enth_mol_comp = Expression(
            self.params.component_list,
            rule=_rule_enth_mol_comp,
        )

    def _entr_mass(self):
        def entr_mass_rule(b):
            return sum (b.entr_mass_comp[i] for i in b.params.component_list)
        self.entr_mass = Expression (rule=entr_mass_rule)

    def _entr_mass_comp(self):
        def _rule_entr_mass_comp(b, i):
            return b.entr_mol_comp[i] /b.params.mw_comp[i]
        self.entr_mass_comp = Expression(
            self.params.component_list,
            rule=_rule_entr_mass_comp,
        )
        
    def _entr_mol_comp(self):
        def _rule_entr_mol_comp(b, i):
            return b.entr_mol * b.mole_frac_comp[i]
        self.entr_mol_comp = Expression(
            self.params.component_list,
            rule=_rule_entr_mol_comp,
        )

    def _flow_mass(self):
        def flow_mass_rule(b):
            return sum(b.flow_mass_comp[i] for i in b.params.component_list)
        self.flow_mass = Expression(rule = flow_mass_rule)
    
    def _flow_mass_comp(self):
        def _rule_flow_mass_comp(b, i):
            return b.flow_mol_comp[i] * b.params.mw_comp[i]
        self.flow_mass_comp = Expression(
            self.params.component_list,
            rule=_rule_flow_mass_comp,
        )

    def _flow_mol_comp(self):
        def _rule_flow_mol_comp(b, i):
            return b.mole_frac_comp[i] * b.flow_mol
        self.flow_mol_comp = Expression(
            self.params.component_list,
            rule=_rule_flow_mol_comp,
      )
    
    def _flow_vol(self):
        def _rule_flow_vol(b):
            return b.vol_mol*b.flow_mol
        self.flow_vol = Expression(rule=_rule_flow_vol)

    def _mass_frac_comp(self):
        def _mass_frac_comp_rule(b, i):
            return b.flow_mol_comp[i] * b.params.mw_comp[i]
        self.mass_frac_comp = Expression ( self.params.component_list, rule = _mass_frac_comp_rule)

    def _total_energy_flow(self):
        def _rule_total_energy_flow(b):
            return b.flow_mass * b.enth_mass
        self.total_energy_flow = Expression( rule = _rule_total_energy_flow)
    
    def get_material_flow_terms(self, p, c):
        return self.mole_frac_comp[c]*self.flow_mol

    def get_enthalpy_flow_terms(self , p):
        return self.flow_mol * self.enth_mol

    def default_material_balance_type(self):
        return MaterialBalanceType.componentTotal

    def default_energy_balance_type(self):
        return EnergyBalanceType.enthalpyTotal

    def define_state_vars(self):
        return {
            "flow_mol": self.flow_mol,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "mole_frac_comp": self.mole_frac_comp
        }




@declare_process_block_class("SurrogateParameterBlock")
class SurrogateParameterData(PhysicalParameterBlock):

    ## called on block construction
    def build(self):

        # Calling parent build method
        super(GenericParameterData, self).build()

        # Setting default units
        self.get_metadata().add_default_units(self.config.base_units)

        # Call configure method to set construction arguments
        self.configure()

        # Define the surrogate block
        self._state_block_class = SurrogateStateBlock # noqa: F821

    def configure(self):
        ## placeholder for configuration
        return None

    def parameters(self):
        ## placeholder for parameters
        return None

