import benchmark

if __name__ == '__main__':
    with open('/root/guohao/pytorch_benchmark/benchmark/configs/models.txt', 'w') as f:
        for md in benchmark.models.load_trad_model.ModelDescriptions:
            f.write(md.value.name + '\n')
