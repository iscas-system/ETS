from objects import ModelType
from .grouping_based_executor import MLPTest_GroupingBasedExecutor, GCNGroupingBasedExecutor
from .op_based_executor import MLP_OPBasedExecutor, PerfNet_OPBasedExecutor, GBDT_OPBasedExecutor
from .subgraph_based_executor import MLPTest_SubgraphBasedExecutor, \
    TransformerSubgraphBasedExecutor, \
    LSTMSubgraphBasedExecutor, \
    GRUSubgraphBasedExecutor, \
    GCNSubgraphBasedExecutor


def get_executor_cls(model_type: ModelType):
    return {
        ModelType.GBDT: GBDT_OPBasedExecutor,
        ModelType.MLP: MLP_OPBasedExecutor,
        ModelType.PerfNet: PerfNet_OPBasedExecutor,
        ModelType.MLPTestGrouping: MLPTest_GroupingBasedExecutor,
        ModelType.MLPTestSubgraph: MLPTest_SubgraphBasedExecutor,
        ModelType.Transformer: TransformerSubgraphBasedExecutor,
        ModelType.LSTM: LSTMSubgraphBasedExecutor,
        ModelType.GRU: GRUSubgraphBasedExecutor,
        ModelType.GCNSubgraph: GCNSubgraphBasedExecutor,
        ModelType.GCNGrouping: GCNGroupingBasedExecutor,
    }[model_type]
