from pyomo.environ import ConcreteModel
from idaes.core import FlowsheetBlock


m = ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)
m.fs.pp = HDAParameterBlock()