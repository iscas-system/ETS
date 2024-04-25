class Pointwise():
    # TODO: Add more operators.
    # Unary
    unary = ["abs", "abs_", "neg", "neg_", "reciprocal", "reciprocal_"]
    unary += ["__abs__", "__neg__"]

    # Unary bitwise
    unary += ["__invert__"]

    # Exponential and log (unary)
    exp_log = ["exp", "exp_", "exp1m", "exp1m_", "log", "log_",
               "log10", "log10_", "log1p", "log1p_", "log2", "log2_"]

    # Sqrt (unary)
    sqrt = ["rsqrt", "rsqrt_", "sqrt", "sqrt_"]

    # Representation (unary)
    representation = ["ceil", "ceil_", "clamp", "clamp_", "floor", "floor_",
                      "frac", "frac_", "round", "round_", "sign", "sign_",
                      "trunc", "trunc_"]

    # Trigonometric and transcendental (unary)
    trig_trans = ["acos", "acos_", "asin", "asin_", "atan", "atan_",
                  "atan2", "atan2_", "cos", "cos_", "cosh", "cosh_",
                  "sin", "sin_", "sinh", "sinh_", "tan", "tan_",
                  "sigmoid", "sigmoid_", "tanh", "tanh_"]

    # Error (unary)
    error = ["erf", "erf_", "erfc", "erfc_", "erfinv", "erfinv_"]

    # Binary
    binary = ["add", "add_", "div", "div_", "mul", "mul_",
              "remainder", "remainder_", "sub", "sub_"]
    binary += ["__add__", "__sub__", "__mul__", "__floordiv__",
               "__truediv__", "__mod__"]
    binary += ["__radd__", "__rsub__", "__rmul__", "__rdiv__",
               "__rtruediv__", "__rfloordiv__"]
    binary += ["fmod", "fmod_"]

    # Binary inplace
    ibinary = ["__iadd__", "__isub__", "__imul__", "__itruediv__"]

    # Power (binary)
    power = ["pow", "pow_", "__pow__", "__rpow__"]

    # Comparison (binary)
    comp = ["lt", "lt_", "gt", "gt_", "ge", "ge_", "le", "le_",
            "eq", "eq_", "ne", "ne_"]
    comp += ["__lt__", "__gt__", "__ge__", "__le__", "__eq__", "__ne__"]

    # Logical (binary)
    logical = ["__and__", "__or__", "__xor__", "__lshift__", "__rshift__"]

    # Logical inplace (binary)
    ilogical = ["__iand__", "__ior__", "__ixor__", "__ilshift__", "__irshift__"]

    # Ternary
    ternary = ["addcdiv", "addcdiv_", "addcmul", "addcmul_"]

    # Misc
    misc = ["digamma", "lerp", "lerp_", "mvlgamma"]

    ops = unary + binary + ibinary + comp + logical + ilogical + \
          ternary + exp_log + power + sqrt + representation + trig_trans + \
          error + misc


class Convert():
    ops = ["byte", "char", "double", "float", "half", "int", "long", "short", "to"]


class OneZero():
    ops = ["ones", "ones_like", "zero_", "zeros", "zeros_like"]


class Activation():
    # ops = [
    #     "celu", "elu", "elu_", "hardshrink", "hardtanh", "hardtanh_", "leaky_relu", "leaky_relu_", "logsigmoid",
    #     "prelu", "relu", "relu_", "relu6", "rrelu", "rrelu_", "selu", "sigmoid", "softplus", "softshrink", "softsign",
    #     "tanh", "tanhshrink", "threshold", "threshold_"
    # ]

    ops = ["threshold", "relu", "rrelu", "hardtanh", "relu6", "sigmoid", "hardsigmoid", "tanh",
           "silu", "mish", "hardswish", "elu", "celu", "selu", "glu", "gelu", "hardshrink", "leaky_relu",
           "logsigmoid", "softplus", "softshrink", "multi_head_attention_forward", "prelu", "softsign", "tanhshrink",
           "softmin", "softmax", "softmax2d", "logsoftmax","DivBackward1"
           ]


class Optimizer():
    ops = [
        "cross_entropy"
    ]


opCodes = {}


class Pool():
    ops = [
        "max_pool1d", "max_pool2d", "max_pool3d", "max_unpool1d", "max_unpool2d", "max_unpool3d",
        "avg_pool1d", "avg_pool2d", "avg_pool3d", "fractional_max_pool2d", "fractional_max_pool3d", "lp_pool1d",
        "lp_pool2d", "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d", "adaptive_avg_pool1d",
        "adaptive_avg_pool2d", "adaptive_avg_pool3d"
    ]


class Normalization():
    ops = [
        "layer_norm", "local_response_norm",
    ]


class Op(object):
    ops = Normalization.ops + Pointwise.ops + Convert.ops + OneZero.ops + Activation.ops + Optimizer.ops + Pool.ops + [
        "linear", "bias","conv2d", "conv1d", "batch_norm",
        "__matmul__", "matmul",
        "embedding", "sum", "mean", "norm", "dropout",
        "cat", "reshape", "masked_scatter_", "gather",
        "nonzero", "index_select", "masked_select",
        "addmm", "addmm_", "mm", "bmm",
        "softmax", "log_softmax", "mse_loss", "adam",
        "randperm","pad",
        "normalize", "roll", "new_zeros", "aten::index", "bernoulli_",  # swin_v2_t
        "fill_", "full", "copy_", "clone", "aten::clone", "contiguous",  # todo aten::clone only after cat
        "any", "to", "hardswish", "hardsigmoid"]


def getOpCodes():
    if (len(opCodes) == 0):
        for i in range(len(Op.ops)):
            opCodes[Op.ops[i]] = int(i)
    #matmul
    # opCodes["matmul"] = opCodes["mul"]
    # opCodes["__matmul__"] = opCodes["mul"]
    # opCodes["__mul__"] = opCodes["mul"]
    # opCodes["mul_"] = opCodes["mul"]
    return opCodes

if __name__ == '__main__':
    with open("config/opCodes.dict", "w") as f:
        f.write(str(getOpCodes()))