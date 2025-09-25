# toluene_surrogate_property.py
import os
from pyomo.environ import Var, Reals, NonNegativeReals, units as pyunits
from idaes.core import (
    PhysicalParameterBlock,
    StateBlock,
    StateBlockData,
    declare_process_block_class,
)
from idaes.surrogate.pysmo import PysmoSurrogate
from idaes.surrogate.surrogate_block import SurrogateBlock


@declare_process_block_class("TolueneParameterBlock")
class TolueneParameterData(PhysicalParameterBlock):
    def build(self):
        super().build()
        # Component + phases
        self.component_list = ["toluene"]
        self.phase_list = ["Liq", "Vap"]

        # Molecular weight [kg/mol]
        self.mw = pyunits.kg / pyunits.mol * 0.09214  # 92.14 g/mol

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties(
            {
                "enth_mol": {"method": "_enth_mol"},
                "entr_mol": {"method": "_entr_mol"},
                "vol_mol": {"method": "_vol_mol"},
                "enth_mass": {"method": "_enth_mass"},
                "vol_mass": {"method": "_vol_mass"},
            }
        )
        obj.add_default_units(
            {
                "time": pyunits.s,
                "length": pyunits.m,
                "mass": pyunits.kg,
                "amount": pyunits.mol,
                "temperature": pyunits.K,
            }
        )


class _TolueneStateBlock(StateBlockData):
    def build(self):
        super().build()
        self._make_state_vars()
        self._build_surrogate()
        self._build_property_relations()

    def _make_state_vars(self):
        # State vars
        self.flow_mol = Var(
            domain=NonNegativeReals,
            initialize=1.0,
            units=pyunits.mol/pyunits.s,
            doc="Total molar flowrate [mol/s]",
        )
        self.temperature = Var(
            domain=NonNegativeReals,
            initialize=350,
            bounds=(200, 600),
            units=pyunits.K,
            doc="Temperature [K]",
        )
        self.pressure = Var(
            domain=NonNegativeReals,
            initialize=101325,
            bounds=(10000, 5000000),
            units=pyunits.Pa,
            doc="Pressure [Pa]",
        )
        self.phase_indicator = Var(
            domain=Reals,
            initialize=0,
            bounds=(0, 1),
            units=pyunits.dimensionless,
            doc="Phase indicator (0=liq, 1=vap)",
        )

        # Surrogate outputs
        self.enth_mol = Var(
            domain=Reals,
            initialize=1000,
            units=pyunits.J/pyunits.mol,
            doc="Molar enthalpy [J/mol]",
        )
        self.entr_mol = Var(
            domain=Reals,
            initialize=10,
            units=pyunits.J/pyunits.mol/pyunits.K,
            doc="Molar entropy [J/mol/K]",
        )
        self.vol_mol = Var(
            domain=NonNegativeReals,
            initialize=1e-4,
            units=pyunits.m**3/pyunits.mol,
            doc="Molar volume [m³/mol]",
        )

    def _build_surrogate(self):
        # Load trained surrogate (example JSON file from PySMO)
        script_dir = os.path.dirname(__file__)
        surrogate_file = os.path.join(script_dir, "toluene_surrogate.json")

        self.pysmo_surrogate = PysmoSurrogate.load_from_file(surrogate_file)

        inputs = [self.temperature, self.pressure, self.phase_indicator]
        outputs = [self.enth_mol, self.entr_mol, self.vol_mol]

        self.surrogate = SurrogateBlock()
        self.surrogate.build_model(
            self.pysmo_surrogate,
            input_vars=inputs,
            output_vars=outputs,
        )

    def _build_property_relations(self):
        mw = self.params().mw

        # Derived properties
        self.enth_mass = self.enth_mol / mw
        self.vol_mass = self.vol_mol / mw

    def _enth_mol(self):
        return self.enth_mol

    def _entr_mol(self):
        return self.entr_mol

    def _vol_mol(self):
        return self.vol_mol

    def _enth_mass(self):
        return self.enth_mass

    def _vol_mass(self):
        return self.vol_mass


class TolueneStateBlock(StateBlock):
    def state_block_class(self):
        return _TolueneStateBlock