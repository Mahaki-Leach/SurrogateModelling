from idaes.models.properties.modular_properties.examples.BT_PR import configuration
from idaes.models.properties.modular_properties.base import GenericParameterBlock
from pyomo.environ import ConcreteModel, FlowsheetBlock

def generate_augmented_BT_PR():
    """
    Generates augmented data for training.
    """

    f = open("output.txt", "a")

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.prop = GenericParameterBlock(**configuration)
    m.fs.prop.build_state_block([1], defined_state=True)

    # Fixing the most common inputs (4 inputs)
    m.fs.prop[1].flow_mol.fix(1)
    m.fs.prop[1].temperature.fix(300)
    m.fs.prop[1].pressure.fix(101325)
    m.fs.prop[1].mole_frac_comp["benzene"].fix(0.5)
    m.fs.prop[1].mole_frac_comp["toluene"].fix(0.5)

    # Getting outputs






    
