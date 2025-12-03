from pyomo.environ import ConcreteModel, value
from PP_v2 import SurrogateParameterBlock, SurrogateStateBlock  # adjust import path
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
from pyomo.environ import ConcreteModel
from idaes.core import FlowsheetBlock

# Load trained surrogates
surr = PysmoSurrogate.load_from_file("Benzene_v2.json")

# Create a model and add the property package
m = ConcreteModel()
m.params = SurrogateParameterBlock(
    surrogate=surr,
    mw=0.07811,
)

# Build a state block
m.props = m.params.build_state_block([1], defined_state=True)

m.props[1].flow_mol.fix(1)  # mol/s
m.props[1].q.fix(1)
m.props[1].temperature.fix(550)  # K
m.props[1].pressure.fix(101325)  # Pa

m.props.initialize(outlvl=1)

m.props.display()