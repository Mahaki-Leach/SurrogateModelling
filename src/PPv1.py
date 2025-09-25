import os
from pyomo.environ import Var, Expression, units as pyunits
from idaes.core import PropertyPackage, StateBlock, StateBlockData
from idaes.core.util.misc import add_object_reference
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate

class MultiPhaseSurrogatePackage(PropertyPackage):
    def __init__(self, liquid_file, vapor_file, twophase_file, **kwargs):
        super().__init__(**kwargs)
        script_dir = os.path.dirname(__file__)

        # Load pre-trained surrogates
        self.liquid_surr = PysmoSurrogate.load_from_file(
            os.path.join(script_dir, liquid_file)
        )
        self.vapor_surr = PysmoSurrogate.load_from_file(
            os.path.join(script_dir, vapor_file)
        )
        self.twophase_surr = PysmoSurrogate.load_from_file(
            os.path.join(script_dir, twophase_file)
        )

    def build_state_block(self, **state_args):
        # Create a StateBlock for each flow state
        return MultiPhaseStateBlock(
            block=None, property_package=self, **state_args
        )


class MultiPhaseStateBlock(StateBlockData):
    def build(self):
        # --- State Variables ---
        self.temperature = Var(initialize=350, units=pyunits.K, doc="Temperature")
        self.pressure = Var(initialize=101325, units=pyunits.Pa, doc="Pressure")
        self.q = Var(initialize=0.0, bounds=(0, 1), doc="Vapor fraction")

        # --- Property Variables ---
        self.enth_mol = Var(initialize=1.0, units=pyunits.J/pyunits.mol, doc="Molar enthalpy")
        self.entr_mol = Var(initialize=1.0, units=pyunits.J/pyunits.mol/pyunits.K, doc="Molar entropy")
        self.vol_mol = Var(initialize=1.0, units=pyunits.m**3/pyunits.mol, doc="Molar volume")

        # Phase-specific properties
        self.enth_mol_liq = Var(initialize=1.0, units=pyunits.J/pyunits.mol, doc="Liquid enthalpy")
        self.enth_mol_vap = Var(initialize=1.0, units=pyunits.J/pyunits.mol, doc="Vapor enthalpy")

        # --- Surrogate Evaluation ---
        self.surrogate_eval = Expression(expr=self._evaluate_surrogate())

    def _evaluate_surrogate(self):
        # Get the current state
        T = self.temperature.value
        P = self.pressure.value
        q_val = self.q.value

        # Determine which surrogate to use
        if q_val == 0.0:
            surr = self.parent_block().liquid_surr
            inputs = [T, P]
        elif q_val == 1.0:
            surr = self.parent_block().vapor_surr
            inputs = [T, P]
        else:
            surr = self.parent_block().twophase_surr
            inputs = [T, P, q_val]

        # Evaluate surrogate
        result = surr(*inputs)  # returns dict with outputs

        # Map outputs to state variables
        self.enth_mol.value = result["enth_mol"]
        self.entr_mol.value = result["entr_mol"]
        self.vol_mol.value = result["vol_mol"]

        # Optionally store phase-specific values (for two-phase)
        if "enth_mol_liq" in result:
            self.enth_mol_liq.value = result["enth_mol_liq"]
        else:
            self.enth_mol_liq.value = self.enth_mol.value

        if "enth_mol_vap" in result:
            self.enth_mol_vap.value = result["enth_mol_vap"]
        else:
            self.enth_mol_vap.value = self.enth_mol.value

        return 0  # Dummy Expression, surrogate side-effect sets variables