from pyomo.environ import ConcreteModel, value
from PP import SurrogateParameterBlock, SurrogateStateBlock  # adjust import path
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
from pyomo.environ import ConcreteModel
from idaes.core import FlowsheetBlock

# Load trained surrogates
liquid_surr = PysmoSurrogate.load_from_file("Benzene_Liquid_v2.json")
vapor_surr = PysmoSurrogate.load_from_file("Benzene_Vapor_v2.json")
twophase_surr = PysmoSurrogate.load_from_file("Benzene_TwoPhase_v2.json")
# Create a model and add the property package

m = ConcreteModel()
m.params = SurrogateParameterBlock(
    liquid_surr=liquid_surr,
    vapor_surr=vapor_surr,
    twophase_surr=twophase_surr
)

# Build a state block
m.props = m.params.build_state_block([1], defined_state=True)

m.props[1].flow_mol.fix(1)  # mol/s
m.props[1].q.fix(0.6330976547750866)  # vapor fraction
m.props[1].temperature.fix(306)  # K
m.props[1].pressure.fix(18191)  # Pa

m.props.initialize(outlvl=1)

m.props.display()



# print(value(m.props[1].enth_mol_expr))
# m.props.display()