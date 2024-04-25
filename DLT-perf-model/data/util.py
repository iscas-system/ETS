op_freq = {}

def get_op_frequency():
    global op_freq
    if len(op_freq) == 0:
        with open("configs/op_frequency.dict", "r") as f:
            op_freq = eval(f.read())
    return op_freq
