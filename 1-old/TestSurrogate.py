# Import Python libraries
import logging
import os

# Import Pyomo libraries
from pyomo.environ import (
    Reference,
    Expression,
    Constraint,
    Param,
    Reals,
    value,
    Var,
    NonNegativeReals,
    units,
)

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    PhysicalParameterBlock,
    StateBlockData,
    StateBlock,
    MaterialBalanceType,
    EnergyBalanceType,
    VaporPhase,
    LiquidPhase,
    Component,
)
from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
import idaes.logger as idaeslog

from pyomo.environ import Block, Constraint
from pyomo.core.base.expression import ScalarExpression, Expression, _GeneralExpressionData, ExpressionData
from pyomo.core.base.var import ScalarVar, _GeneralVarData, VarData, IndexedVar, Var
from pyomo.common.config import ConfigBlock, ConfigDict, ConfigValue, In, Bool
from io import StringIO


@declare_process_block_class("SurrogateParameterBlock")
class PhysicalParameterData(PhysicalParameterBlock):