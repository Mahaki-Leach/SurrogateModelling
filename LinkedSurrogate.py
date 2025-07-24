from pyomo.environ import *
from pyomo.gdp import *

model = ConcreteModel()

# Variables
model.quality = Var(bounds=(0, 1))
model.x = Var()
model.y = Var()

def populate_block(self, block, additional_options=None):
        """Populate a Pyomo Block with surrogate model constraints.

        Args:
            block: Pyomo Block component to be populated with constraints.
            additional_options: None
                No additional options are required for this surrogate object

        Returns:
            None
        """

        output_set = Set(initialize=self._output_labels, ordered=True)

        # Dummy constraints for illustration
        def liquid_rule(b, o):
            in_vars = block.input_vars_as_dict()
            out_vars = block.output_vars_as_dict()
            return out_vars[o] == self._trained.get_result(o).model.generate_expression(
                list(in_vars.values())
            )

        def vapor_rule(b, o):
            in_vars = block.input_vars_as_dict()
            out_vars = block.output_vars_as_dict()
            return out_vars[o] == self._vapor._trained.get_result(o).model.generate_expression(
                list(in_vars.values())
            )

        def two_phase_rule(b, o):
            in_vars = block.input_vars_as_dict()
            out_vars = block.output_vars_as_dict()
            return out_vars[o] == self._trained.get_result(o).model.generate_expression(
                list(in_vars.values())
            )

        model.liquid = Disjunct()
        model.liquid.c = Constraint(output_set, rule=liquid_rule)
        model.vapor = Disjunct()
        model.vapor.c = Constraint(output_set, rule=vapor_rule)
        model.two_phase = Disjunct()
        model.two_phase.c = Constraint(output_set, rule=two_phase_rule)

        # Define logic on quality
        model.phase_selector = Disjunction(expr=[
            (model.quality == 0, model.liquid),
            (model.quality == 1, model.vapor),
            (inequality(0, model.quality, 1), model.two_phase),
        ])

        # Required transformation
        TransformationFactory('gdp.bigm').apply_to(model)