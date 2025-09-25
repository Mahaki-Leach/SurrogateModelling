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

        # --- Liquid surrogate ---
        self.liquid_surr = SurrogatePropertyBlock(
            surrogate=self.params.config.liquid_surr,
            input_vars=["T_in","P_in"],
            output_vars=["enth_out","entr_out","z_out"]
        )

        self.vapor_surr = SurrogatePropertyBlock(
            surrogate=self.params.config.vapor_surr,
            input_vars=["T_in","P_in"],
            output_vars=["enth_out","entr_out","z_out"]
        )

        self.twophase_surr = SurrogatePropertyBlock(
            surrogate=self.params.config.twophase_surr,
            input_vars=["T_in","P_in","q_in"],
            output_vars=["enth_out","entr_out"]
        )

        # -------------------------------------------------------
        # Phase indicator variables
        # -------------------------------------------------------
        # self.is_liq = Var(domain=Binary)
        # self.is_vap = Var(domain=Binary)
        # self.is_mix = Var(domain=Binary)

        # # Only one phase active
        # self.single_phase = Constraint(expr=self.is_liq + self.is_vap + self.is_mix == 1)

        # # -------------------------------------------------------
        # # Connect inputs (always active)
        # # -------------------------------------------------------
        # self.liq_T_link = Constraint(expr=self.surrogates.liquid.inputs["T_in"] == self.temperature)
        # self.liq_P_link = Constraint(expr=self.surrogates.liquid.inputs["P_in"] == self.pressure)

        # self.vap_T_link = Constraint(expr=self.surrogates.vapor.inputs["T_in"] == self.temperature)
        # self.vap_P_link = Constraint(expr=self.surrogates.vapor.inputs["P_in"] == self.pressure)

        # self.mix_T_link = Constraint(expr=self.surrogates.twophase.inputs["T_in"] == self.temperature)
        # self.mix_P_link = Constraint(expr=self.surrogates.twophase.inputs["P_in"] == self.pressure)
        # self.mix_q_link = Constraint(expr=self.surrogates.twophase.inputs["q_in"] == self.q)

        # # -------------------------------------------------------
        # # Blend surrogate outputs with indicators
        # # -------------------------------------------------------
        # self.enth_link = Constraint(
        #     expr=self.enth_mol ==
        #         self.is_liq * self.surrogates.liquid.outputs["enth_out"] +
        #         self.is_vap * self.surrogates.vapor.outputs["enth_out"] +
        #         self.is_mix * self.surrogates.twophase.outputs["enth_out"]
        # )

        # self.entr_link = Constraint(
        #     expr=self.entr_mol ==
        #         self.is_liq * self.surrogates.liquid.outputs["entr_out"] +
        #         self.is_vap * self.surrogates.vapor.outputs["entr_out"] +
        #         self.is_mix * self.surrogates.twophase.outputs["entr_out"]
        # )

        # # Optional: composition "z"
        # self.z_link = Constraint(
        #     expr=self.z ==
        #         self.is_liq * self.surrogates.liquid.outputs["z_out"] +
        #         self.is_vap * self.surrogates.vapor.outputs["z_out"]
        #         # no z for two-phase in this sketch
        # )

        # Molar volume
        # R = self.params.R
        # def vol_mol_rule(b):
        #     return b.z * R * b.temperature / b.pressure
        # self.vol_mol = Expression(rule=vol_mol_rule, doc="Molar volume [m3/mol]")

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

@declare_process_block_class("SurrogateParameterBlock")
class SurrogateParameterData(PhysicalParameterBlock):

    CONFIG = PhysicalParameterBlock.CONFIG()

    CONFIG.declare("liquid_surr", ConfigValue(default=None))
    CONFIG.declare("vapor_surr", ConfigValue(default=None))
    CONFIG.declare("twophase_surr", ConfigValue(default=None))

    def build(self):
      super(SurrogateParameterData, self).build()

      self._state_block_class = SurrogateStateBlock

      # Components
      self.benzene = Component()
      self.Liq = LiquidPhase()
      self.Vap = VaporPhase()

      # Parameters
      self.R = Param(initialize=8.314, units=pyunits.J/pyunits.mol/pyunits.K, mutable=True)
      self.temperature_ref = Param(mutable=True, default=298.15, units=pyunits.K)
      self.pressure_ref = Param(mutable=True, default=101325, units=pyunits.Pa)

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties(
            {
                "flow_mol": {"method": None, "units": units.mol / units.s},
                "pressure": {"method": None, "units": units.Pa},
                "enth_mol": {"method": None, "units": units.J / units.mol / units.K},
                "entr_mol": {"method": None, "units": units.J / units.mol / units.K},
                "temperature": {"method": None, "units": units.K},
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



class SurrogatePropertyBlock(Block):
    """
      Wraps a single SurrogateBlock as an IDAES-compatible property block
    """
    def __init__(self, surrogate, input_vars, output_vars, **kwargs):
        super().__init__(**kwargs)
        self.surrogate = surrogate
        # State variables used as inputs
        self.T = Var(domain=Reals, initialize=300)
        self.P = Var(domain=Reals, initialize=1e5)
        if "q_in" in input_vars:
            self.q = Var(domain=Reals, bounds=(0,1), initialize=0.0)
        # Output placeholders
        for out in output_vars:
            setattr(self, out, Var(domain=Reals))

    def update_outputs(self):
        """Evaluate surrogate numerically (for initialization or calculation)"""
        inputs = {"T": value(self.T), "P": value(self.P)}
        if hasattr(self, "q"):
            inputs["q"] = value(self.q)
        results = self.surrogate.predict(**inputs)
        for key, val in results.items():
            if hasattr(self, key):
                getattr(self, key).value = val