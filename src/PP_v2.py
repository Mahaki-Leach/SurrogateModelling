from pyomo.environ import Var, Reals, NonNegativeReals, Expression, value
from pyomo.core.expr.numeric_expr import Expr_ifExpression as if_expression
from idaes.core import StateBlockData
from pyomo.environ import Param, Block, units as pyunits

from pyomo.environ import Var, Reals, NonNegativeReals, Expression, units
from idaes.core import StateBlockData
from idaes.core.surrogate.surrogate_block import SurrogateBlock

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
    PhaseType,
)

from pyomo.environ import Var, Binary, Constraint, Block

from idaes.core.solvers import get_solver
from idaes.core.util.initialization import (fix_state_vars,
                                            revert_state_vars,
                                            solve_indexed_blocks)
from idaes.core.util.model_statistics import degrees_of_freedom, \
                                             number_unfixed_variables
from idaes.core.util.constants import Constants as const
import idaes.logger as idaeslog

from idaes.core import PhysicalParameterBlock, declare_process_block_class
from pyomo.common.config import ConfigValue
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate

from pyomo.environ import Set, Param, value

from idaes.core import StateBlock

def prepare_state(blk, state_args, state_vars_fixed):
    # Fix state variables if not already fixed
    if state_vars_fixed is False:
        flags = fix_state_vars(blk, state_args)
    else:
        flags = None

    # Deactivate sum of mole fractions constraint
    for k in blk.keys():
        if blk[k].config.defined_state is False:
            blk[k].mole_fraction_constraint.deactivate()

    # Check that degrees of freedom are zero after fixing state vars
    for k in blk.keys():
        if degrees_of_freedom(blk[k]) != 0:
            raise Exception("State vars fixed but degrees of freedom "
                            "for state block is not zero during "
                            "initialization.")

    return flags

def unfix_state(blk, flags, outlvl):
    init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="properties")

    # Reactivate sum of mole fractions constraint
    for k in blk.keys():
        if blk[k].config.defined_state is False:
            blk[k].mole_fraction_constraint.activate()

    if flags is not None:
        # Unfix state variables
        revert_state_vars(blk, flags)

    init_log.info_high("State Released.")

def initialize_state(blk, solver, init_log, solve_log):
    # Check that there is something to solve for
    free_vars = 0
    for k in blk.keys():
        free_vars += number_unfixed_variables(blk[k])
    if free_vars > 0:
        # If there are free variables, call the solver to initialize
        try:
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = solve_indexed_blocks(solver, [blk], tee=True)#slc.tee)
        except:
            res = None
    else:
        res = None

    init_log.info("Properties Initialized {}.".format(
        idaeslog.condition(res)))

def unfix_state(blk, flags, outlvl):
    init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="properties")

    # Reactivate sum of mole fractions constraint
    for k in blk.keys():
        if blk[k].config.defined_state is False:
            blk[k].mole_fraction_constraint.activate()

    if flags is not None:
        # Unfix state variables
        revert_state_vars(blk, flags)

    init_log.info_high("State Released.")

def restore_state(blk, flags, hold_state):
    # Return state to initial conditions
    if hold_state is True:
        return flags
    else:
        blk.release_state(flags)

class _SurrogateStateBlock(StateBlock):

    def initialize(blk, state_args=None, state_vars_fixed=False,
                    hold_state=False, outlvl=idaeslog.NOTSET,
                    solver=None, optarg=None):

        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="properties")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="properties")

        # Create solver
        solver_obj = get_solver(solver, optarg)

        flags = prepare_state(blk, state_args, state_vars_fixed)

        initialize_state(blk, solver_obj, init_log, solve_log)
        restore_state(blk, flags, hold_state)

        init_log.info("Initialization Complete")

    def release_state(blk, flags, outlvl=idaeslog.NOTSET):
        unfix_state(blk, flags, outlvl)

@declare_process_block_class("SurrogateStateBlock", block_class=_SurrogateStateBlock)
class SurrogateStateBlockData(StateBlockData):

    """

    StateBlockData for SurrogatePropertyPackage using IDAES SurrogateBlock.

    Uses 3 surrogates:
        - liquid_surr: inputs T, P -> outputs enth_mol, entr_mol, z
        - vapor_surr: inputs T, P -> outputs enth_mol, entr_mol, z
        - twophase_surr: inputs T, P, q -> outputs enth_mol, entr_mol

    """

    def build(self):
        super().build()

        # State variables
        self.flow_mol = Var(domain=NonNegativeReals, initialize=1.0, units=pyunits.mol/pyunits.s)
        self.temperature = Var(domain=NonNegativeReals, initialize=300, units=pyunits.K)
        self.pressure = Var(domain=NonNegativeReals, initialize=101325, units=pyunits.Pa)
        self.q = Var(domain=Reals, bounds=(0,1), initialize=0.0, units=pyunits.dimensionless)

        self.mw = Param(initialize=self.params.mw, units=pyunits.kg/pyunits.mol)

        self.z = Var(domain=NonNegativeReals, initialize=0.5, bounds=(0,1), units=pyunits.dimensionless)
        self.enth_mol = Var(domain=Reals, initialize=30000, units=pyunits.J/pyunits.mol)
        self.entr_mol = Var(domain=Reals, initialize=100, units=pyunits.J/pyunits.mol/pyunits.K)
        self.vol_mol = Var(domain=Reals, initialize=0.0001, units=pyunits.m**3/pyunits.mol)

        inputs = [self.temperature, self.pressure, self.q]
        outputs = [self.enth_mol, self.entr_mol, self.vol_mol]

        self.surrogate = SurrogateBlock()

        self.surrogate.build_model(
            self.params.config.surrogate,
            input_vars=inputs,
            output_vars=outputs,
        )

    # Derived Thermodynamic properties
    # -----------------------------

    def _flow_mol_comp(self):
        def rule_flow_mol_comp(b, i):
            return b.mole_frac_comp[i] * b.flow_mol
        self.flow_mol_comp = Expression(
            self.params.component_list,
            rule=rule_flow_mol_comp,
        )

    def _flow_mass(self):
        def rule_flow_mass(b):
            return b.flow_mol * b.mw
        self.flow_mass = Expression(rule=rule_flow_mass)

    def _flow_mass_comp(self):
        def rule_flow_mass_comp(b, i):
            return b.flow_mol_comp[i] * b.params.mw_comp[i]
        self.flow_mass_comp = Expression(
            self.params.component_list,
            rule=rule_flow_mass_comp,
        )

    def _flow_vol(self):
        def rule_flow_vol(b):
            return b.flow_mol * b.vol_mol
        self.flow_vol = Expression(rule=rule_flow_vol)

    def _flow_mol_phase(self):
        def rule(b, p):
            if p == "Vap":
                return b.flow_mol * b.q
            elif p == "Liq":
                return b.flow_mol * (1 - b.q)
        self.flow_mol_phase = Expression(self.params.phase_list, rule=rule)


    def _flow_mass_phase(self):
        def rule(b, p):
            return b.flow_mol_phase[p] * b.mw
        self.flow_mass_phase = Expression(self.params.phase_list, rule=rule)


    def _flow_mol_phase_comp(self):
        def rule(b, p, i):
            return b.mole_frac_phase_comp[p, i] * b.flow_mol_phase[p]
        self.flow_mol_phase_comp = Expression(
            self.params.phase_list,
            self.params.component_list,
            rule=rule,
        )

    def _flow_mass_phase_comp(self):
        def rule(b, p, i):
            return b.flow_mass_phase[p] * b.mass_frac_phase_comp[p, i]
        self.flow_mass_phase_comp = Expression(
            self.params.phase_list,
            self.params.component_list,
            rule=rule,
        )

    def _mole_frac_comp(self):
        def rule(b, i):
            return b.flow_mol_comp[i] / b.flow_mol
        self.mole_frac_comp = Expression(
            self.params.component_list,
            rule=rule,
        )

    def _mass_frac_comp(self):
        def rule(b, i):
            return b.flow_mass_comp[i] / b.flow_mass
        self.mass_frac_comp = Expression(
            self.params.component_list,
            rule=rule,
        )

    def _enth_mol_phase(self):
        def rule(b, p):
            return b.enth_mol
        self.enth_mol_phase = Expression(self.params.phase_list, rule=rule)


    def _enth_mass_phase(self):
        def rule(b, p):
            return b.enth_mol_phase[p] / b.mw
        self.enth_mass_phase = Expression(self.params.phase_list, rule=rule)


    def _enth_mass(self):
        def rule(b):
            return b.enth_mol / b.mw
        self.enth_mass = Expression(rule=rule)


    def _enth_mass_comp(self):
        def rule(b, i):
            return b.enth_mass * b.mass_frac_comp[i]
        self.enth_mass_comp = Expression(self.params.component_list, rule=rule)


    def _enth_mol_comp(self):
        def rule(b, i):
            return b.enth_mol * b.mole_frac_comp[i]
        self.enth_mol_comp = Expression(self.params.component_list, rule=rule)


    def _enth_mol_phase_comp(self):
        def rule(b, p, i):
            return b.enth_mol_phase[p] * b.mole_frac_phase_comp[p, i]
        self.enth_mol_phase_comp = Expression(
            self.params.phase_list,
            self.params.component_list,
            rule=rule,
        )

    def _entr_mol_phase(self):
        def rule(b, p):
            return b.entr_mol
        self.entr_mol_phase = Expression(self.params.phase_list, rule=rule)


    def _entr_mass_phase(self):
        def rule(b, p):
            return b.entr_mol_phase[p] / b.mw
        self.entr_mass_phase = Expression(self.params.phase_list, rule=rule)


    def _entr_mass(self):
        def rule(b):
            return b.entr_mol / b.mw
        self.entr_mass = Expression(rule=rule)


    def _entr_mol_comp(self):
        def rule(b, i):
            return b.entr_mol * b.mole_frac_comp[i]
        self.entr_mol_comp = Expression(self.params.component_list, rule=rule)


    def _entr_mass_comp(self):
        def rule(b, i):
            return b.entr_mass * b.mass_frac_comp[i]
        self.entr_mass_comp = Expression(self.params.component_list, rule=rule)


    def _entr_mol_phase_comp(self):
        def rule(b, p, i):
            return b.entr_mol_phase[p] * b.mole_frac_phase_comp[p, i]
        self.entr_mol_phase_comp = Expression(
            self.params.phase_list,
            self.params.component_list,
            rule=rule,
        )

    def _entr_mass_phase_comp(self):
        def rule(b, p, i):
            return b.entr_mass_phase[p] * b.mass_frac_phase_comp[p, i]
        self.entr_mass_phase_comp = Expression(
            self.params.phase_list,
            self.params.component_list,
            rule=rule,
        )

    # -----------------------------

    # Material and energy flow terms

    # -----------------------------

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
            "q": self.q
        }

    # -----------------------------

@declare_process_block_class("SurrogateParameterBlock")
class SurrogateParameterData(PhysicalParameterBlock):

    CONFIG = PhysicalParameterBlock.CONFIG()

    CONFIG.declare("surrogate", ConfigValue(default=None))
    CONFIG.declare("mw", ConfigValue(default=0.07811, domain=float))

    def build(self):
      super(SurrogateParameterData, self).build()

      self._state_block_class = SurrogateStateBlock

      # Components
      self.benzene = Component()

      # Phases
      self.Liq = LiquidPhase()
      self.Vap = VaporPhase()

      # Parameters
      # self.R = Param(initialize=8.314, units=pyunits.J/pyunits.mol/pyunits.K, mutable=True)
      self.temperature_ref = Param(mutable=True, default=298.15, units=pyunits.K)
      self.pressure_ref = Param(mutable=True, default=101325, units=pyunits.Pa)

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties(
            {
                "flow_mol": {"method": None, "units": units.mol / units.s},
                "q": {"method": None, "units": units.dimensionless},
                "flow_mol_comp": {"method": "_flow_mol_comp"},
                "flow_mass": {"method": "_flow_mass", "units": units.kg / units.s},
                "flow_mass_comp": {"method": "_flow_mass_comp"},
                "flow_vol": {"method":"_flow_vol", "units": units.m**3 / units.s},
                "pressure": {"method": None, "units": units.Pa},
                "enth_mol_phase": {"method": "_enth_mol_phase"},
                "enth_mass_phase": {"method": "_enth_mass_phase"},
                "enth_mass": {"method": "_enth_mass", "units": units.J/units.kg},
                "enth_mass_comp": {"method": "_enth_mass_comp"},
                "mole_frac_comp": {"method": "_mole_frac_comp"},
                "mass_frac_comp": {"method": "_mass_frac_comp"},
                "entr_mol": {"method": None, "units": units.J / units.mol / units.K},
                "entr_mol_phase": {"method": "_entr_mol_phase"},
                "entr_mass_phase": {"method": "_entr_mass_phase"},
                "entr_mass": {"method": "_entr_mass", "units": units.J / units.kg / units.K},
                "temperature": {"method": None, "units": units.K},
                "flow_mass_phase": {"method": "_flow_mass_phase"},
                "flow_mol_phase": {"method": "_flow_mol_phase"},
                "mole_frac_phase_comp": {"method": "_mole_frac_phase_comp"},
                "mass_frac_phase_comp": {"method": "_mass_frac_phase_comp"},
                "flow_mass_phase": {"method": "_flow_mass_phase"},
                "flow_mol_phase": {"method": "_flow_mol_phase"},
                "flow_mass_phase_comp": {"method": "_flow_mass_phase_comp"},
                "flow_mol_phase_comp": {"method": "_flow_mol_phase_comp"},
                "entr_mol_comp": {"method": "_entr_mol_comp"},
                "entr_mass_comp": {"method": "_entr_mass_comp"},
                "entr_mol_phase_comp": {"method": "_entr_mol_phase_comp"},
                "entr_mass_phase_comp": {"method": "_entr_mass_phase_comp"},
                "enth_mol_comp": {"method": "_enth_mol_comp"},
                "enth_mass_comp": {"method": "_enth_mass_comp"},
                "enth_mol_phase_comp": {"method": "_enth_mol_phase_comp"},
                "enth_mass_phase_comp": {"method": "_enth_mass_phase_comp"},
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



# class SurrogatePropertyBlock(Block):
#     """
#       Wraps a single SurrogateBlock as an IDAES-compatible property block
#     """
#     def __init__(self, surrogate, input_vars, output_vars, **kwargs):
#         super().__init__(**kwargs)
#         self.surrogate = surrogate
#         # State variables used as inputs
#         self.T = Var(domain=Reals, initialize=300)
#         self.P = Var(domain=Reals, initialize=1e5)
#         if "q_in" in input_vars:
#             self.q = Var(domain=Reals, bounds=(0,1), initialize=0.0)
#         # Output placeholders
#         for out in output_vars:
#             setattr(self, out, Var(domain=Reals))

#     def update_outputs(self):
#         """Evaluate surrogate numerically (for initialization or calculation)"""
#         inputs = {"T": value(self.T), "P": value(self.P)}
#         if hasattr(self, "q"):
#             inputs["q"] = value(self.q)
#         results = self.surrogate.predict(**inputs)
#         for key, val in results.items():
#             if hasattr(self, key):
#                 getattr(self, key).value = val