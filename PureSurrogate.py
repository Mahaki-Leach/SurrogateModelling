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
    LiquidPhase,
    Component,
)
from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
import idaes.logger as idaeslog

from pyomo.environ import Block, Constraint
from pyomo.core.base.expression import ScalarExpression, Expression, _GeneralExpressionData, ExpressionData
from pyomo.core.base.var import ScalarVar, _GeneralVarData, VarData, IndexedVar, Var
from pyomo.common.config import ConfigBlock, ConfigDict, ConfigValue, In, Bool
from io import StringIO

# Set up logger
_log = logging.getLogger(__name__)

class _StateBlock(StateBlock):
    """
    This Class contains methods which should be applied to Property Blocks as a
    whole, rather than individual elements of indexed Property Blocks.
    """

    def initialize(
        blk,
        state_args=None,
        hold_state=False,
        outlvl=1,
        state_vars_fixed=False,
        solver="ipopt",
        optarg={"tol": 1e-8},
    ):
        # Fix state vars
        # Release state vars
        
        if state_vars_fixed is False:
            # Fix state variables if not already fixed
            Fcflag = {}
            Pflag = {}
            Tflag = {}
            # Fmflag = {}

            for k in blk.keys():
                # if blk[k].mole_frac_comp["water"].fixed is True:
                #     Fmflag[k] = True
                # else:
                #     Fmflag[k] = False
                #     if state_args is None:
                #         blk[k].mole_frac_comp["water"].fix()
                #     else:
                #         blk[k].mole_frac_comp["water"].fix(state_args["flow_mol_water"])

                if blk[k].flow_mol.fixed is True:
                    Fcflag[k] = True
                else:
                    Fcflag[k] = False
                    if state_args is None:
                        blk[k].flow_mol.fix()
                    else:
                        blk[k].flow_mol.fix(state_args["flow_mol"])

                if blk[k].pressure.fixed is True:
                    Pflag[k] = True
                else:
                    Pflag[k] = False
                    if state_args is None:
                        blk[k].pressure.fix()
                    else:
                        blk[k].pressure.fix(state_args["pressure"])

                if blk[k].temperature.fixed is True:
                    Tflag[k] = True
                else:
                    Tflag[k] = False
                    if state_args is None:
                        blk[k].temperature.fix()
                    else:
                        blk[k].temperature.fix(state_args["temperature"])

            # If input block, return flags, else release state
            flags = {"Fcflag": Fcflag, "Pflag": Pflag, "Tflag": Tflag}

        else:
            # Check when the state vars are fixed already result in dof 0
            for k in blk.keys():
                if degrees_of_freedom(blk[k]) != 0:
                    raise Exception(
                        "State vars fixed but degrees of freedom "
                        "for state block is not zero during "
                        "initialization."
                    )

        if state_vars_fixed is False:
            if hold_state is True:
                return flags
            else:
                blk.release_state(flags)

    def release_state(blk, flags, outlvl=0):
        if flags is None:
            return

        # Unfix state variables
        for k in blk.keys():
            if flags["Fcflag"][k] is False:
                blk[k].flow_mol.unfix()
            if flags["Pflag"][k] is False:
                blk[k].pressure.unfix()
            if flags["Tflag"][k] is False:
                blk[k].temperature.unfix()
            # if flags["Fmflag"][k] is False:
            #     blk[k].mole_frac_comp["water"].unfix()

        if outlvl > 0:
            if outlvl > 0:
                _log.info("{} State Released.".format(blk.name))


@declare_process_block_class("SurrogateStateBlock", block_class=_StateBlock)
class SurrogateStateBlockData(StateBlockData):
    """

    """

    def build(self):
        """
        Callable method for Block construction
        """

        # Links parameters to self.params
        super(StateBlockData, self).build()

        # Other setup
        self.constraints = Block()
        self._make_state_vars()
    

    def initialize(
        self,
        state_args=None,
        hold_state=False,
        outlvl=idaeslog.NOTSET,
        state_vars_fixed=False,
        solver="ipopt",
        optarg={"tol": 1e-8},
    ):
        
        # Setup loggers
        init_log = idaeslog.getInitLogger(
            self.name, self.config.output_level, tag="properties"
        )
        solve_log = idaeslog.getSolveLogger(
            self.name, self.config.output_level, tag="properties"
        )

        # Create solver object
        solver_obj = get_solver(
            solver=self.config.solver,
            solver_options=self.config.solver_options,
            writer_config=self.config.solver_writer_config,
        )

        # 
        # Adding custom property packages
        self.vapor_phase = Block()
        self.liquid_phase = Block()

        # Enable equilibrium calculation


        init_log.info("Starting initialization routine")

        ### Initialize the state block

        # Check if state is within VLE bounds
        # Determine phase
        # Check if state is within VLE bounds


    def _make_state_vars(self):
        
        # TODO: Initialisation values
        # TODO: Check bounds
        # TODO: Check input / output
        
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
            bounds=(193.15, 1000),
            units=units.K,
            doc="Temperature [K]",
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


        # Building input & output lists
        input_list = []
        output_list = []

        for i in self.params.config.input_labels:
            comp = self.find_component(i)
            if comp is None:
                raise ValueError(
                    f"Input component {i} not found in the parameter block."
                )
            self.input_list.append(i)
        
        for o in self.params.config.output_labels:
            comp = self.find_component(o)
            if comp is None:
                raise ValueError(
                    f"Output component {o} not found in the parameter block."
                )
            self.output_list.append(o)

        # Creating surrogate from JSON 
        self.pysmo_surrogate = PysmoSurrogate.load(
            StringIO(self.params.surrogate)
        )

        # Building surrogate block on model
        self.surrogate = SurrogateBlock()
        self.surrogate.build_model(
            self.pysmo_surrogate,
            input_vars=input_list,
            output_vars=output_list,
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

    def _enth_mol_comp(self): ###check this
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
    
    # def _vapor_frac(self):
    #     self.vapor_frac = Var(
    #         domain = NonNegativeReals,
    #         initialize=1.0,
    #     )

    def _x_property(self, p):
        if p == LiquidPhase:
            return self.liquid_phase._x_property(p)
        elif p == VaporPhase:
            return self.vapor_phase._x_property(p)

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

    def model_check(blk):
        """
        Model checks for property block
        """
        # Check temperature bounds
        if value(blk.temperature) < blk.temperature.lb:
            _log.error("{} temperature set below lower bound.".format(blk.name))
        if value(blk.temperature) > blk.temperature.ub:
            _log.error("{} temperature set above upper bound.".format(blk.name))

        # Check pressure bounds
        if value(blk.pressure) < blk.pressure.lb:
            _log.error("{} Pressure set below lower bound.".format(blk.name))
        if value(blk.pressure) > blk.pressure.ub:
            _log.error("{} Pressure set above upper bound.".format(blk.name))


@declare_process_block_class("SurrogateParameterBlock")
class PhysicalParameterData(PhysicalParameterBlock):
    """
    Defines global parameters and components for the Surrogate Property Package.
    """

    ## Creating configuration object populated with ppb parameters
    cfg = PhysicalParameterBlock.CONFIG()

    cfg.declare(
        "components",
        ConfigValue(
            default=dict,
        )
    )

    cfg.declare(
        "surrogate",
        ConfigValue(
            default=dict,
        )
    )

    cfg.declare(
        "bounds",
        ConfigValue(
            default=dict,
        )
    )

    #
    # FPHx 
    # 
    # VLE: {
    #     "temperature": (193.15),
    #     "pressure": (10000),
    #     "enthalpy": (0, 1000000)
    # }
    #
    #


    def build(self):
        """
        Callable method for Block construction.
        """

        super(PhysicalParameterData, self).build()
        self._state_block_class = SurrogateStateBlock # noqa: F821

        # Building components
        for c in self.config.components:
            self.add_component(
                c, Component()
            )
        
        # Building both phases
        self.Vap = VaporPhase()
        self.Liq = LiquidPhase()

        # need to populate component_list?


        ## getting bounds from cfg



    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties(
            {
                # Potential Inputs / Outputs
                "temperature": {"method": None, "units": units.K},
                "flow_mol": {"method": None, "units": units.mol / units.s},
                "pressure": {"method": None, "units": units.Pa},
                "vol_mol" : {"method": None, "units": units.m**3 / units.mol},
                "enth_mol": {"method": None, "units": units.J / units.mol},
                "entr_mol": {"method": None, "units": units.J / units.mol / units.K},
                
                # Derived properties
                "flow_mol_comp": {"method": "_flow_mol_comp"},
                "flow_mass": {"method": "_flow_mass", "units": units.kg / units.s},
                "flow_mass_comp": {"method": "_flow_mass_comp"},
                "flow_vol": {"method":"_flow_vol", "units": units.m**3 / units.s},
                "vol_mass": {"method": "_vol_mass", "units": units.m**3 / units.kg},
                "enth_mol_comp": {"method": "_enth_mol_comp"},
                "enth_mass": {"method": "_enth_mass", "units": units.J/units.kg},
                "enth_mass_comp": {"method": "_enth_mass_comp"},
                "mole_frac_comp": {"method": "_mole_frac_comp"},
                "mass_frac_comp": {"method": "_mass_frac_comp"},
                "entr_mol_comp": {"method": "_entr_mol_comp"},
                "entr_mass": {"method": "_entr_mass", "units": units.J / units.kg / units.K},
                "entr_mass_comp": {"method": "_entr_mass_comp"},
                "total_energy_flow": {"method": "_total_energy_flow", "units": units.kW},
            }
        )

        obj.add_default_units(
            {
                "time": units.s,
                "length": units.m,
                "mass": units.kg,
                "amount": units.mol,
                "temperature": units.K,
            }
        )