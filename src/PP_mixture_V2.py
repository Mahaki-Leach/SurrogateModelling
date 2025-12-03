# Mixture Property Package


from pyomo.environ import Var, Reals, NonNegativeReals, Expression, value
from pyomo.core.expr.numeric_expr import Expr_ifExpression as if_expression
from idaes.core import StateBlockData
from pyomo.environ import Param, Block, units as pyunits

from pyomo.environ import Var, Reals, NonNegativeReals, Expression, units
from idaes.core import StateBlockData
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from PP_v2 import SurrogateParameterBlock

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
from pyomo.core import Piecewise

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

        # Call mixture-specific init here if it exists
        for k in blk.keys():
            if hasattr(blk[k], "_mixture_post_init"):
                blk[k]._mixture_post_init()

        restore_state(blk, flags, hold_state)

        init_log.info("Initialization Complete")

    def release_state(blk, flags, outlvl=idaeslog.NOTSET):
        unfix_state(blk, flags, outlvl)

@declare_process_block_class("SurrogateMixtureStateBlock", block_class=_SurrogateStateBlock)
class SurrogateMixtureStateBlockData(StateBlockData):

    """

    StateBlockData for Mixture using IDAES SurrogateBlock.

    """

    def _mixture_post_init(self):
        q_val = value(self.q)

        if q_val == 1:
            self.benzene_quality.fix(1)
            self.toluene_quality.fix(1)
        else:
            self.benzene_quality.fix((self.c1_vap*self.q)/self.mole_frac_comp["benzene"])
            self.toluene_quality.fix((self.c2_vap*self.q)/self.mole_frac_comp["toluene"])

        self.benzene_pure_sb[0].q.fix(self.benzene_quality)
        self.benzene_pure_sb[0].temperature.fix(self.temperature)
        self.benzene_pure_sb[0].pressure.fix(self.pressure)

        self.benzene_pure_sb.initialize()

        self.toluene_pure_sb[0].q.fix(self.toluene_quality)
        self.toluene_pure_sb[0].temperature.fix(self.temperature)
        self.toluene_pure_sb[0].pressure.fix(self.pressure)

        self.toluene_pure_sb.initialize()

    def build(self):
        super().build()

        # State variables
        self.flow_mol = Var(domain=NonNegativeReals, initialize=1.0, units=pyunits.mol/pyunits.s)
        self.temperature = Var(domain=NonNegativeReals, initialize=300, units=pyunits.K)
        self.pressure = Var(domain=NonNegativeReals, initialize=101325, units=pyunits.Pa)
        self.mole_frac_comp = Var(
            self.params.component_list,
            domain=NonNegativeReals,
            initialize=0.5,
            units=pyunits.dimensionless,
        )

        self.c1_liq = Var(bounds=(0,1), initialize=0.5)
        self.c1_vap = Var(bounds=(0,1), initialize=0.5)
        self.q = Var(bounds=(0,1), initialize=0.5)

        # Deriving other properties

        self.c2_liq = Expression(expr=1 - self.c1_liq)
        self.c2_vap = Expression(expr=1 - self.c1_vap)

        inputs = [self.temperature, self.pressure, self.mole_frac_comp["benzene"]]
        outputs = [self.q, self.c1_liq, self.c1_vap]

        self.surrogate = SurrogateBlock()

        self.surrogate.build_model(
            PysmoSurrogate.load_from_file("BT_Q_5K_W_Q.json"),
            input_vars=inputs,
            output_vars=outputs,
        )

        # Building child pure surrogate state blocks

        self.benzene_pure_sb = self.params.benzene_pure.build_state_block([0], defined_state=True)
        self.toluene_pure_sb = self.params.toluene_pure.build_state_block([0], defined_state=True)

        self.benzene_quality = Var(bounds=(0,1), initialize=0.5)
        self.toluene_quality = Var(bounds=(0,1), initialize=0.5)

        self.benzene_pure_sb[0].temperature.fix(self.temperature)
        self.benzene_pure_sb[0].pressure.fix(self.pressure)
        self.benzene_pure_sb[0].q.fix(self.benzene_quality)

        self.toluene_pure_sb[0].temperature.fix(self.temperature)
        self.toluene_pure_sb[0].pressure.fix(self.pressure)
        self.toluene_pure_sb[0].q.fix(self.toluene_quality)

        # Derived Thermodynamic properties
        # -----------------------------

        # Mixture enthalpy (molar) as sum of component contributions
        def _enth_mol_rule(b):
            return b.benzene_pure_sb[0].enth_mol * b.mole_frac_comp["benzene"] \
                + b.toluene_pure_sb[0].enth_mol * b.mole_frac_comp["toluene"]
        self.enth_mol = Expression(rule=_enth_mol_rule)

        # Mixture entropy (molar) as sum of component contributions
        def _entr_mol_rule(b):
            return b.benzene_pure_sb[0].entr_mol * b.mole_frac_comp["benzene"] \
                + b.toluene_pure_sb[0].entr_mol * b.mole_frac_comp["toluene"]
        self.entr_mol = Expression(rule=_entr_mol_rule)

        # Mixture molar volume as weighted sum
        def _vol_mol_rule(b):
            return b.benzene_pure_sb[0].vol_mol * b.mole_frac_comp["benzene"] \
                + b.toluene_pure_sb[0].vol_mol * b.mole_frac_comp["toluene"]
        self.vol_mol = Expression(rule=_vol_mol_rule)

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
            "mole_frac_comp": self.mole_frac_comp
        }

    # -----------------------------

@declare_process_block_class("SurrogateMixtureParameterBlock")
class SurrogateMixtureParameterData(PhysicalParameterBlock):

    CONFIG = PhysicalParameterBlock.CONFIG()

    def build(self):
        super(SurrogateMixtureParameterData, self).build()

        self._state_block_class = SurrogateMixtureStateBlock

        # Components
        self.benzene = Component()
        self.toluene = Component()

        # Phases
        self.Liq = LiquidPhase()
        self.Vap = VaporPhase()

        # Associated pure surrogate packages
        surr1 = PysmoSurrogate.load_from_file("Benzene.json")
        surr2 = PysmoSurrogate.load_from_file("Toluene.json")

        self.benzene_pure = SurrogateParameterBlock(surrogate=surr1, mw=0.07811)
        self.toluene_pure = SurrogateParameterBlock(surrogate=surr2, mw=0.09214)

        # Parameters
        self.temperature_ref = Param(mutable=True, default=298.15, units=pyunits.K)
        self.pressure_ref = Param(mutable=True, default=101325, units=pyunits.Pa)

    @classmethod
    def define_metadata(cls, obj):

        # obj.add_properties({
        #     "enth_mol": {"method": "_enth_mol", "units": units.J/units.mol},
        # })

        obj.add_default_units(
            {
                "time": units.s,
                "length": units.m,
                "mass": units.kg,
                "amount": units.mol,
                "temperature": units.K,
            }
        )