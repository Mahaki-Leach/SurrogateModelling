import os
from pyomo.environ import Var, NonNegativeReals, Reals, units, Expression, value
from idaes.core import PhysicalParameterBlock, StateBlock, StateBlockData, MaterialFlowBasis
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
from pyomo.common.config import ConfigValue


class SurrogatePropertyPackage(PhysicalParameterBlock):

    cfg = PhysicalParameterBlock.CONFIG()

    cfg.declare(
        "liquid_surr",
        ConfigValue(
            default=dict,
        ))
    cfg.declare(
        "vapor_surr",
        ConfigValue(
            default=dict,
        ))
    cfg.declare(
        "twophase_surr",
        ConfigValue(
            default=dict,
        ))

    def build(self):
        super().build()
        # Store surrogates as attributes
        self.liquid_surr = self.config.liquid_surr
        self.vapor_surr = self.config.vapor_surr
        self.twophase_surr = self.config.twophase_surr

        # Could define global constants, e.g., R
        self.R = 8.314

    def build_state_block(self, **kwargs):
        return SurrogateStateBlock(self, **kwargs)


class SurrogateStateBlock(StateBlock):
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)
        self.params = params
        # State variables
        self.temperature = Var(domain=NonNegativeReals, initialize=300)
        self.pressure = Var(domain=NonNegativeReals, initialize=1e5)
        self.q = Var(domain=(0,1), initialize=0.0)
        self.flow_mol = Var(domain=NonNegativeReals, initialize=1.0)

        # Expressions
        self.enth_mol = Expression(rule=self._enth_rule)
        self.entr_mol = Expression(rule=self._entr_rule)
        self.z = Expression(rule=self._z_rule)
        self.vol_mol = Expression(rule=self._vol_rule)

    # Expression rules
    def _enth_rule(self, b):
        T = value(self.temperature)
        P = value(self.pressure)
        q = value(self.q)
        if q <= 0.0:
            return self.params.liquid_surr.predict([T,P])[0]
        elif q >= 1.0:
            return self.params.vapor_surr.predict([T,P])[0]
        else:
            return self.params.twophase_surr.predict([T,P,q])[0]

    def _entr_rule(self, b):
        T = value(self.temperature)
        P = value(self.pressure)
        q = value(self.q)
        if q <= 0.0:
            return self.params.liquid_surr.predict([T,P])[1]
        elif q >= 1.0:
            return self.params.vapor_surr.predict([T,P])[1]
        else:
            return self.params.twophase_surr.predict([T,P,q])[1]

    def _z_rule(self, b):
        T = value(self.temperature)
        P = value(self.pressure)
        q = value(self.q)
        if q <= 0.0:
            return self.params.liquid_surr.predict([T,P])[2]
        elif q >= 1.0:
            return self.params.vapor_surr.predict([T,P])[2]
        else:
            Z_liq = self.params.liquid_surr.predict([T,P])[2]
            Z_vap = self.params.vapor_surr.predict([T,P])[2]
            return (1-q)*Z_liq + q*Z_vap

    def _vol_rule(self, b):
        R = 8.314
        return value(self.z) * R * value(self.temperature) / value(self.pressure)