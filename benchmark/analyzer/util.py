import logging


class InputType(object):
    _types = {
        "<class 'torch.FloatTensor'> ": 1,
        "<class 'torch.HalfTensor'> ": 2,
        "<class 'torch.DoubleTensor'> ": 3,
        "<class 'torch.FloatTensor'>": 1,
        "<class 'torch.HalfTensor'>": 2,
        "<class 'torch.DoubleTensor'>": 3,
    }


class Dtype(object):
    _types = {
        "uint8": (1, "uint8"),
        "int8": (1, "int8"),
        "byte": (1, "byte"),
        "char": (1, "char"),
        "bool": (1, "bool"),

        "int16": (2, "int16"),
        "int32": (4, "int32"),
        "int64": (8, "int64"),

        "fp16": (2, "float16"),
        "fp32": (4, "float32"),
        "fp64": (8, "float64"),
    }


def parseModelInfo(info: str):
    # model=alexnet, batch=32, hw=[256, 256], dtype=<class 'torch.FloatTensor'> iter=0
    info_dic = {}
    # info_dic['model'] = info.split(',')[0].split('=')[1]
    info_dic['batch'] = str(info.split(',')[1].split('=')[1])
    # info_dic['h'] = str(info.split(',')[2].split('[')[1])
    # info_dic['w'] = str(info.split(',')[3].split(']')[0])
    info_dic['h'] = str(info.split(',')[2].split('=')[1])
    info_dic['w'] = str(info.split(',')[3].split('=')[1])

    info_dic['input_type'] = info.split(',')[4].split('=')[1].split('i')[0]
    info_dic['input_type'] = InputType._types[info_dic['input_type']]
    # info_dic['iter'] = str(info.split(',')[4].split('=')[2])
    return info_dic


def parse_grid(grid: str):
    """
    解析grid
    :param grid:
    :return:
    """
    # grid = grid[1:-1]
    grid = grid.split(',')
    res = 1
    for g in grid:
        res *= int(g)
    return res


def parse_block(block: str):
    """
    解析block
    :param block:
    :return:
    """
    # block = block[1:-1]
    block = block.split(',')
    res = 1
    for b in block:
        res *= int(b)
    return res


def parse_param(param: str):
    if param.startswith('na='):
        return [], 0
    res = [0 for i in range(30)]
    if param.endswith(','):
        param = param[0:len(param) - 1]

    # 等号的参数解析
    if param.find('=') != -1:
        idx = 0
        if param.find(',') == -1:
            param = [param]
        else:
            param = param.split(',')
        for info in param:
            info = info.split('=')
            if info[0] == 'type':
                # res[idx] += Dtype._types[info[1]][0]
                continue
            else:
                res[idx] = int(info[1])
            idx += 1
        return res, idx
    else:
        for t in Dtype._types.keys():
            if param.endswith(t):
                param = param[0:len(param) - len(t)]
                # res[0] = Dtype._types[t][0]  # data size
        if param.endswith(','):
            param = param[0:len(param) - 1]
        idx = 0
        if param == '[]':
            return res, idx
        elif param[0] == '[':  # 解析[1，2，3，4数组],可以转化为成绩
            param = param[1:-1]
            # 若不含,则无法split
            if param.find(',') == -1:
                param = [param]
            else:
                param = param.split(',')
            for info in param:
                res[idx] = int(info)
                idx += 1
        else:
            # print('parse param error: %s' % param)
            logging.warning('parse param error: %s' % param)
            return [], 0  # 都不符合
    return res, idx


def parse_params(params: str):
    """
    解析params
    :param params:
    :return:
    """
    # 只有一个参数，长为30
    # print('parse parameters:', params)
    if ';' not in params:
        tmp, idx = parse_param(params)
        if idx < 30:
            tmp = tmp[0:idx] + [0 for i in range(30 - idx)]
        return tmp
    # 两个参数，各自占15
    if params.endswith(';'):
        params = params[0:len(params) - 1]
    params = params.split(';')
    num = len(params)
    res = []
    for param in params:
        tmp, idx = parse_param(param)
        if idx > 30 // num:
            #todo: multi_head_attention_forward有7个参数，无法实现对齐
            logging.warning('parse params error: %s' % params)
            tmp = tmp[0:idx]
        else:
            tmp = tmp[0:idx] + [0 for i in range(30 // num - idx)]
        res += tmp
    if len(res) < 30:
        res = [0 for i in range(30 - len(res))]
    return res

if __name__ == '__main__':
    print(parse_params('N=2,C=3,H=32,W=32,K=64,P=16,Q=16,R=7,S=7,ph=3,pw=3,U=2,V=2,dh=1,dw=1,g=1,type=fp16,'))
    print(parse_params('[]int32;[]int32'))
    print(parse_params('[2,64,16,16]fp16'))
    print(parse_params('[143]fp16;[]fp32'))
    print(parse_params('[512,512,3,3]fp16;[512,512,3,3]fp16'))

