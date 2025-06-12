from GenericSurrogate import SurrogateParameterBlock
from pyomo.environ import ConcreteModel, Var, value
from idaes.core import FlowsheetBlock

m = ConcreteModel()
m.fs = FlowsheetBlock()
m.fs.properties = SurrogateParameterBlock()
m.fs.properties.build_state_block([1], defined_state=True)

m.fs.properties[1].temperature.fix(300)
m.fs.properties[1].pressure.fix(101325)
m.fs.properties[1].mole_frac_comp["benzene"].fix(0.5)
m.fs.properties[1].mole_frac_comp["toluene"].fix(0.5)

m.fs.initialize()
