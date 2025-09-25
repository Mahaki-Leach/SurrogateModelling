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
    PhaseType,
)
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.math import smooth_max, smooth_min
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
import idaes.logger as idaeslog

from pyomo.environ import Block, Constraint
from pyomo.core.base.expression import ScalarExpression, Expression, _GeneralExpressionData, ExpressionData
from pyomo.core.base.var import ScalarVar, _GeneralVarData, VarData, IndexedVar, Var
from idaes.models.properties.general_helmholtz import (
    HelmholtzParameterBlock,
    PhaseType as helmPhaseType,
    StateVars,
)



# Some more information about this module
__author__ = "Ben Lincoln & Stephen Burroughs"


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
        """
        Initialisation routine for property package.

        Keyword Arguments:
            flow_mol : value at which to initialize component flows
                             (default=None)
            pressure : value at which to initialize pressure (default=None)
            temperature : value at which to initialize temperature
            mole_flow_frac: value at which to initialise the molar flow fraction
                          (default=None)
            outlvl : sets output level of initialisation routine

                     * 0 = no output (default)
                     * 1 = return solver state for each step in routine
                     * 2 = include solver output information (tee=True)
            state_vars_fixed: Flag to denote if state vars have already been
                              fixed.
                              - True - states have already been fixed by the
                                       control volume 1D. Control volume 0D
                                       does not fix the state vars, so will
                                       be False if this state block is used
                                       with 0D blocks.
                             - False - states have not been fixed. The state
                                       block will deal with fixing/unfixing.
            optarg : solver options dictionary object (default=None)
            solver : str indicating which solver to use during
                     initialization (default = 'ipopt')
            hold_state : flag indicating whether the initialization routine
                         should unfix any state variables fixed during
                         initialization (default=False).
                         - True - states variables are not unfixed, and
                                 a dict of returned containing flags for
                                 which states were fixed during
                                 initialization.
                        - False - state variables are unfixed after
                                 initialization by calling the
                                 release_state method

        Returns:
            If hold_states is True, returns a dict containing flags for
            which states were fixed during initialization.
        """

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
        """
        Method to release state variables fixed during initialisation.

        Keyword Arguments:
            flags : dict containing information of which state variables
                    were fixed during initialization, and should now be
                    unfixed. This dict is returned by initialize if
                    hold_state=True.
            outlvl : sets output level of of logging
        """
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

class _StateBlockWrapper(_StateBlock):

    def initialize(blk, *args, **kwargs):
        for v, k in blk.items():
            print(k)
            k.constraints.deactivate()
        return _StateBlock.initialize(blk, *args, **kwargs)

    def release_state(blk, flags, outlvl=idaeslog.NOTSET):
        _StateBlock.release_state(blk, flags, outlvl)

        for v, k in blk.items():
            k.constraints.activate()

@declare_process_block_class("HAirStateBlock", block_class=_StateBlockWrapper)
class HAirStateBlockData(StateBlockData):
    """
    An example property package for ideal gas properties with Gibbs energy
    """

    def build(self):
        """
        Callable method for Block construction
        """
        super(HAirStateBlockData, self).build()
        self.constraints = Block()
        self._make_state_vars()

        # if self.config.defined_state is False:
        #     self.sum_mole_frac_out = Constraint(
        #         expr = 1.0 == sum(self.mole_frac_comp[i] for i in self.component_list)
        #     )

    def constrain_component(blk, component: Var | Expression, value: float) -> Constraint | Var | None:
        """
        Constrain a component to a value
        """
        if type(component) == ScalarExpression:
            c = Constraint(expr=component == value)
            c.defining_state_var = True
            blk.constraints.add_component(component.local_name, c)
            return c
        elif type(component) in (ScalarVar, _GeneralVarData, VarData, IndexedVar):
            component.fix(value)
            return component
        elif type(component) in (_GeneralExpressionData, ExpressionData):
            # allowed, but we don't need to fix it (eg. mole_frac_comp in helmholtz)
            return None
        else:
            raise Exception(
                f"Component {component} is not a Var or Expression: {type(component)}"
            )

    def _make_state_vars(self):

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
        self.mole_frac_vap_sat = Var(
            domain=Reals,
            initialize=0.03,
            bounds = (0,1),
            units=units.dimensionless,
            doc = "Saturation Mole fraction of water in vapor phase"
        )

        self.enth_mol_vap = Var(
            domain=Reals,
            initialize=300,
            units = units.J / units.mol,
            doc = "Enthalpy [J/mol]"
        )

        self.entr_mol_vap = Var(
            domain=Reals,
            initialize=40,
            units = units.J / units.mol / units.K,
            doc = "Entropy [J/mol/K]"
        )

        self.vol_mol_vap = Var(
            domain = NonNegativeReals,
            initialize=40,
            units = units.m**3 / units.mol
        )

        inputs = [self.temperature, self.pressure, self.mole_frac_comp["water"]]
        outputs = [self.mole_frac_vap_sat, self.enth_mol_vap, self.entr_mol_vap, self.vol_mol_vap]
        script_dir = os.path.dirname(__file__)
        self.pysmo_surrogate = PysmoSurrogate.load_from_file(
            os.path.join(script_dir,"rbf_HA_10000.json")
        )
        self.surrogate = SurrogateBlock()
        self.surrogate.build_model(
            self.pysmo_surrogate,
            input_vars=inputs,
            output_vars=outputs,
        )

        self.water_props = HelmholtzParameterBlock(
        pure_component="h2o", phase_presentation=helmPhaseType.L,state_vars=StateVars.TPX
        )
        # call the methods that have to be constructed automatically.
        self._enth_mol()
        self._entr_mol()

    def _flow_mass(self):
        def _flow_mass_rule(b):
            return b.flow_mol * b.ave_MW
        self.flow_mass = Expression(rule=_flow_mass_rule)

    def _mass_frac_comp(self):
        def _rule_mass_frac_comp(b, i):
            return b.flow_mass_comp[i] / b.flow_mass
        self.mass_frac_comp = Expression(
            self.params.component_list,
            rule=_rule_mass_frac_comp,
        )

    def _ave_MW(self):
        def _rule_ave_MW(b):
            return sum(
                b.mole_frac_comp[i] * b.params.mw_comp[i]
                for i in b.params.component_list
            )
        self.ave_MW = Expression(rule=_rule_ave_MW)

    def _phase_frac_liq(self):
        def _rule_phase_frac_liq(b):
            return smooth_max( b.mole_frac_comp["water"] - b.mole_frac_vap_sat, 0.0)
        self.phase_frac_liq = Expression(rule=_rule_phase_frac_liq)

    def _phase_frac_vap(self):
        def _rule_phase_frac_vap(b):
            return 1 - b.phase_frac_liq

        self.phase_frac_vap = Expression(rule=_rule_phase_frac_vap)

    def _mole_frac_phase_comp(self):
        def _rule_mole_frac_phase_comp(b, p, i):
            if p == "Liq" and i == "water":
                return 1
            elif p == "Liq" and i == "air":
                return 0
            elif p == "Vap":
                xw_vap = smooth_min(b.mole_frac_vap_sat, b.mole_frac_comp["water"])
                total = xw_vap + b.mole_frac_comp["air"]
                if i == "water":
                    return xw_vap / total
                elif i == "air":
                    return b.mole_frac_comp["air"] / total


        self.mole_frac_phase_comp = Expression(
            self.params.phase_list,
            self.params.component_list,
            rule=_rule_mole_frac_phase_comp,
        )

    def _flow_mass_phase_comp(self):
        def _rule_flow_mass_phase_comp(b, p, i):
            return b.mole_frac_phase_comp[p, i] * b.flow_mol_phase[p] * b.params.mw_comp[i]

        self.flow_mass_phase_comp = Expression(
            self.params.phase_list,
            self.params.component_list,
            rule=_rule_flow_mass_phase_comp,
        )
    def _mass_frac_phase_comp(self):
        def _rule_mass_frac_phase_comp(b, p, i):
            return  b.flow_mass_phase_comp[p, i] / b.flow_mass_phase[p]

        self.mass_frac_phase_comp = Expression(
            self.params.phase_list,
            self.params.component_list,
            rule=_rule_mass_frac_phase_comp,
        )


    def _ave_MW_vap(self):
        def _rule_ave_MW_vap(b):
            return sum(
                b.mole_frac_phase_comp["Vap", i] * b.params.mw_comp[i]
                for i in b.params.component_list
            )
        self.ave_MW_vap = Expression(rule=_rule_ave_MW_vap)

    def _enth_mass_phase(self):
        # Nayar et al. (2016), eq. 25 and 26, 0-120 C
        def _rule_enth_mass_phase(b, p):
            # temperature in degC, but units in K
            t = b.temperature - 273.15 * units.K
            P = b.pressure - 101325 * units.Pa
            P_MPa = units.convert(P, to_units=units.MPa)

            # water enthalpy without pressure effects
            h_w = (
                b.params.enth_mass_param_C1
                + b.params.enth_mass_param_C2 * t
                + b.params.enth_mass_param_C3 * t**2
                + b.params.enth_mass_param_C4 * t**3
            )

            # water enthalpy with pressure effects
            h_w_P = h_w + P_MPa * (
                b.params.enth_mass_param_A1
                + b.params.enth_mass_param_A2 * t
                + b.params.enth_mass_param_A3 * t**2
                + b.params.enth_mass_param_A4 * t**3
            )

            if p == "Liq":
                return h_w_P #Todo check reference enthalpy
            elif p == "Vap":
                return b.enth_mol_vap / b.ave_MW_vap #Todo check reference enthalpy

        self.enth_mass_phase = Expression(
            self.params.phase_list,
            rule=_rule_enth_mass_phase,
        )

    def _enth_mass(self):
        def _enth_mass_rule(b):
            return b.enth_mol * b.ave_MW
        self.enth_mass = Expression (rule=_enth_mass_rule)

    def _enth_mass_comp(self):
        def enth_mass_comp_rule(b, i):
            return b.enth_mass * b.mass_frac_comp[i]

        self.enth_mass_comp = Expression(
            self.params.component_list,
            rule=enth_mass_comp_rule,
        )

    def _enth_mass_phase_comp(self):
        def enth_mass_phase_comp_rule(b, p, i):
            return b.enth_mass_phase[p] * b.mass_frac_phase_comp[p, i]
        self.enth_mass_phase_comp = Expression(
            self.params.phase_list,
            self.params.component_list,
            rule=enth_mass_phase_comp_rule,
        )

    def _enth_mol_phase(self):
        def enth_mol_phase_rule(b, p):
            if p == "Liq":
                return b.enth_mass_phase[p] * b.params.mw_comp["water"]
            elif p == "Vap":
                return b.enth_mol_vap

        self.enth_mol_phase = Expression(
            self.params.phase_list,
            rule=enth_mol_phase_rule,
        )

    def _enth_mol(self):
        def enth_mol_rule(b):
            return (b.enth_mol_phase["Liq"] * b.phase_frac_liq + b.enth_mol_phase["Vap"] * b.phase_frac_vap)
        self.enth_mol = Expression (rule=enth_mol_rule)

    def _enth_mol_comp(self):
        def enth_mol_comp_rule(b, i):
            return b.enth_mol * b.mole_frac_comp[i]
        self.enth_mol_comp = Expression(
            self.params.component_list,
            rule=enth_mol_comp_rule,
        )

    def _enth_mol_phase_comp(self):
        def enth_mol_phase_comp_rule(b, p, i):
            return b.enth_mol_phase[p] * b.mole_frac_phase_comp[p, i]
        self.enth_mol_phase_comp = Expression(
            self.params.phase_list,
            self.params.component_list,
            rule=enth_mol_phase_comp_rule,
        )

    def _entr_mol_phase(self):
        def _rule_entr_mol_phase(b, p):
            if p == "Liq":
                return b.water_props.stpx(T=b.temperature,p = b.pressure) #Todo check reference enthalpy
            elif p == "Vap":
                return b.entr_mol_vap #Todo check reference enthalpy
        self.entr_mol_phase = Expression(
            self.params.phase_list,
            rule=_rule_entr_mol_phase,
        )

    def _entr_mass_phase(self):
        def _rule_entr_mass_phase(b, p):
            if p == "Liq":
                return b.enth_mass_phase[p] * b.params.mw_comp["water"]
            elif p == "Vap":
                return b.entr_mol_vap * b.ave_MW_vap
        # Todo check reference enthalpy

        self.entr_mass_phase = Expression(
            self.params.phase_list,
            rule=_rule_entr_mass_phase,
        )

    def _entr_mol(self):
        def _rule_entr_mol(b):
            return (b.entr_mol_phase["Liq"] * b.phase_frac_liq + b.entr_mol_phase["Vap"] * b.phase_frac_vap)
        self.entr_mol = Expression (rule=_rule_entr_mol)

    def _entr_mol_comp(self):
        def _rule_entr_mol_comp(b, i):
            return b.entr_mol * b.mole_frac_comp[i]
        self.entr_mol_comp = Expression(
            self.params.component_list,
            rule=_rule_entr_mol_comp,
        )
    def _entr_mol_phase_comp(self):
        def _rule_entr_mol_phase_comp(b, p, i):
            return b.entr_mol_phase[p] * b.mole_frac_phase_comp[p, i]
        self.entr_mol_phase_comp = Expression(
            self.params.phase_list,
            self.params.component_list,
            rule=_rule_entr_mol_phase_comp,
        )

    def _entr_mass(self):
        def _rule_entr_mass(b):
            return b.entr_mol * b.ave_MW
        self.entr_mass = Expression (rule=_rule_entr_mass)

    def _entr_mass_comp(self):
        def _rule_entr_mass_comp(b, i):
            return b.entr_mass * b.mass_frac_comp[i]
        self.entr_mass_comp = Expression(
            self.params.component_list,
            rule=_rule_entr_mass_comp,
        )
    def _entr_mass_phase_comp(self):
        def _rule_entr_mass_phase_comp(b, p, i):
            return b.entr_mass_phase[p] * b.mass_frac_phase_comp[p, i]
        self.entr_mass_phase_comp = Expression(
            self.params.phase_list,
            self.params.component_list,
            rule=_rule_entr_mass_phase_comp,
        )


    def _flow_mass_comp(self):
        def _rule_flow_mass_comp(b, i):
            return b.flow_mol_comp[i] * b.params.mw_comp[i]
        self.flow_mass_comp = Expression(
            self.params.component_list,
            rule=_rule_flow_mass_comp,
        )

    def _flow_mol_phase(self):
        def _rule_flow_mol_phase(b, p):
            if p == "Liq":
                return b.flow_mol * b.phase_frac_liq
            elif p == "Vap":
                return b.flow_mol * b.phase_frac_vap
        self.flow_mol_phase = Expression(
            self.params.phase_list,
            rule=_rule_flow_mol_phase,
        )

    def _flow_mass_phase(self):
        def _rule_flow_mass_phase(b, p):
            if p == "Liq":
                return b.flow_mol_phase[p] * b.params.mw_comp["water"]
            elif p == "Vap":
                return b.flow_mol_phase[p] * b.ave_MW_vap
        self.flow_mass_phase = Expression(
            self.params.phase_list,
            rule=_rule_flow_mass_phase,
        )

    def _spec_vol_mol_phase(self):
        def _rule_spec_vol_phase(b, p):
            if p == "Liq":
                return 1.80e-5 * units.m**3 / units.mol #assume constant for now
            elif p == "Vap":
                return b.vol_mol_vap
        self.spec_vol_mol_phase = Expression(
            self.params.phase_list,
            rule=_rule_spec_vol_phase,
        )

    def _spec_vol_mass_phase(self):
        def _rule_spec_vol_mass_phase(b, p):
            if p == "Liq":
                return b.spec_vol_mol_phase[p] / b.params.mw_comp["water"]
            elif p == "Vap":
                return b.spec_vol_mol_phase[p] / b.ave_MW_vap
        self.spec_vol_mass_phase = Expression(
            self.params.phase_list,
            rule=_rule_spec_vol_mass_phase,
        )

    def _flow_vol(self):
        def _rule_flow_vol(b):
            return (b.flow_mol * b.phase_frac_vap * b.vol_mol_vap +
                       b.flow_mol * b.phase_frac_liq * b.spec_vol_mol_phase["Liq"])
        self.flow_vol = Expression(rule=_rule_flow_vol)

    def _flow_mol_comp(self):
        def _rule_flow_mol_comp(b, i):
            return b.mole_frac_comp[i] * b.flow_mol
        self.flow_mol_comp = Expression(
            self.params.component_list,
            rule=_rule_flow_mol_comp,
      )

    def _vol_mass_vap(self):
        def _vol_mass_vap_rule(b):
            return b.vol_mol_vap / b.ave_MW_vap
        self.vol_mass_vap = Expression(rule=_vol_mass_vap_rule)

    def _flow_vol_vap(self):
        def _rule_flow_vol(b):
            return b.vol_mol_vap*b.flow_mol
        self.flow_vol = Expression(rule=_rule_flow_vol)

    def _total_energy_flow(self):
        def _rule_total_energy_flow(b):
            return b.flow_mass * b.enth_mass
        self.total_energy_flow = Expression( rule = _rule_total_energy_flow)

    def _vapor_frac(self):
        def _rule_vapor_frac(b):
            return b.phase_frac_vap
        self.vapor_frac = Var(
            domain = NonNegativeReals,
            initialize=1.0,
        )

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


@declare_process_block_class("HAirParameterBlock")
class PhysicalParameterData(PhysicalParameterBlock):
    """
    Property Parameter Block Class

    Contains parameters and indexing sets associated with properties for
    supercritical Humid Air

    """

    def build(self):
        """
        Callable method for Block construction.
        """
        super(PhysicalParameterData, self).build()

        self._state_block_class = HAirStateBlock # noqa: F821

        self.water = Component(valid_phase_types=[PhaseType.liquidPhase, PhaseType.vaporPhase]) #Restrict water to only be in liquid and vapor phases
        self.air = Component(valid_phase_types=[PhaseType.vaporPhase])

        # List of valid phases in property package
        self.Vap = VaporPhase(component_list=[self.water, self.air])
        self.Liq = LiquidPhase(component_list=[self.water])

        # Component list - a list of component identifiers

        mw_comp_dict = {"water":0.01801528, "air": 0.02896}
        self.mw_comp = Param(
            self.component_list,
            mutable=False,
            initialize=mw_comp_dict,
            doc="Molecular weights of components [kg/mol]",
            units=units.kg / units.mol,
        )
        # specific enthalpy parameters, 0-120 C
        # Table 9 in Nayar et al. (2016) - ignores salinity-based parameters
        enth_mass_units = units.J / units.kg
        P_inv_units = units.MPa**-1
        t_inv_units = units.K**-1

        self.enth_mass_param_A1 = Var(
            within=Reals,
            initialize=996.7767,
            units=enth_mass_units * P_inv_units,
            doc="Specific enthalpy parameter A1",
        )
        self.enth_mass_param_A2 = Var(
            within=Reals,
            initialize=-3.2406,
            units=enth_mass_units * P_inv_units * t_inv_units,
            doc="Specific enthalpy parameter A2",
        )
        self.enth_mass_param_A3 = Var(
            within=Reals,
            initialize=0.0127,
            units=enth_mass_units * P_inv_units * t_inv_units**2,
            doc="Specific enthalpy parameter A3",
        )
        self.enth_mass_param_A4 = Var(
            within=Reals,
            initialize=-4.7723e-5,
            units=enth_mass_units * P_inv_units * t_inv_units**3,
            doc="Specific enthalpy parameter A4",
        )
        self.enth_mass_param_C1 = Var(
            within=Reals,
            initialize=141.355,
            units=enth_mass_units,
            doc="Specific enthalpy parameter C1",
        )
        self.enth_mass_param_C2 = Var(
            within=Reals,
            initialize=4202.07,
            units=enth_mass_units * t_inv_units,
            doc="Specific enthalpy parameter C2",
        )
        self.enth_mass_param_C3 = Var(
            within=Reals,
            initialize=-0.535,
            units=enth_mass_units * t_inv_units**2,
            doc="Specific enthalpy parameter C3",
        )
        self.enth_mass_param_C4 = Var(
            within=Reals,
            initialize=0.004,
            units=enth_mass_units * t_inv_units**3,
            doc="Specific enthalpy parameter C4",
        )

        # Specific entropy parameters, 0-120 C


    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties(
            {
                "flow_mol": {"method": None, "units": units.mol / units.s},
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

        obj.define_custom_properties(
            {
                "ave_MW": {"method": "_ave_MW", "units": units.kg / units.mol},
                "ave_MW_vap": {"method": "_ave_MW_vap", "units": units.kg / units.mol},
                "spec_vol_mol_phase": {"method": "_spec_vol_mol_phase"},
                "spec_vol_mass_phase": {"method": "_spec_vol_mass_phase"},
                "phase_frac_liq": {"method": "_phase_frac_liq"},
                "phase_frac_vap": {"method": "_phase_frac_vap"},
                "vol_mol_vap" : {"method": None, "units": units.m**3 / units.mol},
                "vapor_frac": {"method": "_vapor_frac"},
                "flow_vol_vap": {"method": "_flow_vol_vap", "units": units.m**3 / units.s},
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


