from pyomo.environ import ConcreteModel, value
from PP_mixture_V2 import SurrogateMixtureParameterBlock
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
from pyomo.environ import ConcreteModel
from idaes.core import FlowsheetBlock
from pyomo.environ import SolverFactory

solver = SolverFactory('ipopt')


# Create a model and add the property package
m = ConcreteModel()
m.params = SurrogateMixtureParameterBlock()

# Build a state block
# Works in two phase region
# 368, 101325, 0.5, 0.5

m.props = m.params.build_state_block([1], defined_state=True)

m.props[1].flow_mol.fix(1)
m.props[1].temperature.fix(380)
m.props[1].pressure.fix(101325)

m.props[1].mole_frac_comp["benzene"].fix(0.5)
m.props[1].mole_frac_comp["toluene"].fix(0.5)

m.props.initialize(outlvl=6)

# q, enth, entr
print(value(m.props[1].q))
print(value(m.props[1].enth_mol))
print(value(m.props[1].entr_mol))




