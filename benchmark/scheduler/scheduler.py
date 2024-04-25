import json
import pathlib
import random
from typing import List, Dict

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
task_path = BASE_DIR / "scheduler" / "tasks.json"


class GPU:
    def __init__(self, id, type):
        self.id = id
        self.queue = []
        self.type = type
        self.free_time = 0

    def __repr__(self):
        return f"GPU({self.id}, {self.type})"

    def __str__(self):
        return f"GPU({self.id}, {self.type})"

    # to json
    def toJSON(self):
        return {
            "id": self.id,
            "type": self.type,
            "queue": [{
                "id": task.id,
                "info": task.info,
                'time': task.time_info[self.type]
            } for task in self.queue],
        }


class Task:
    def __init__(self, id: str, info: dict, time_info: dict):
        self.id = id
        self.time_info = time_info
        self.info = info  # meta info
        self.status = 0  # 0: not scheduled, 1: scheduled, 2: finished
        self.gpu = None

    def __repr__(self):
        return f"Task({self.id}, {self.info})"

    def __str__(self):
        return f"Task({self.id}, {self.info})"


def JCT(gpus: List[GPU]):
    """Calculate the JCT of the gpus"""
    jct = sum([gpu.free_time for gpu in gpus])
    return jct


def MAKESPAN(gpus: List[GPU]):
    """Calculate the makespan of the gpus"""
    makespan = max([gpu.free_time for gpu in gpus])
    return makespan


def SJF(tasks: Dict, gpus: List[GPU]) -> List[GPU]:
    """Shortest Job First Scheduling Algorithm"""
    # Assign tasks to gpus
    while len(tasks) > 0:
        # Find the shortest task
        shortest_task = None
        end_time = 1000000000
        schelued_gpu = None
        for id, task in tasks.items():
            for gpu in gpus:
                if task.status == 0 and task.time_info[gpu.type] + gpu.free_time < end_time:
                    end_time = task.time_info[gpu.type] + gpu.free_time
                    shortest_task = task
                    schelued_gpu = gpu
        # Assign the shortest task to the gpu
        schelued_gpu.queue.append(shortest_task)
        schelued_gpu.free_time += shortest_task.time_info[schelued_gpu.type]
        shortest_task.status = 1
        tasks.pop(shortest_task.id)
    return gpus


def load_tasks(time_type='measure') -> Dict:
    with open(task_path, 'r') as f:
        tasks = json.load(f)
    res: Dict = {}
    for i in range(len(tasks)):
        id = str(i)
        info = {
            'model': tasks[i]['model'],
            'batch': tasks[i]['batch'],
            'input_size': tasks[i]['h'],
            'dtype': 'float'
        }
        if time_type == 'random':
            time_info = {
                'T4': tasks[i]['T4CPUALL']['measure'] / 1000,
                'P4': tasks[i]['P4CPUALL']['measure'] / 1000,
                '2080Ti': tasks[i]['2080TiCPUALL']['measure'] / 1000,
                '3080Ti': tasks[i]['3080TiCPUALL']['measure'] / 1000
            }
            for key, value in time_info.items():
                time_info[key] = value * 1 + value * 0.5 * random.random()
        else:
            time_info = {
                'T4': tasks[i]['T4CPUALL'][time_type] / 1000,
                'P4': tasks[i]['P4CPUALL'][time_type] / 1000,
                '2080Ti': tasks[i]['2080TiCPUALL'][time_type] / 1000,
                '3080Ti': tasks[i]['3080TiCPUALL'][time_type] / 1000
            }
        res[id] = Task(id, info, time_info)
    return res


def generate_gpus() -> List[GPU]:
    gpus = []
    # for i in range(5):
    #     gpus.append(GPU(str(i), 'T4'))
    for i in range(5, 10):
        gpus.append(GPU(str(i), 'P4'))
    for i in range(10, 15):
        gpus.append(GPU(str(i), '2080Ti'))
    for i in range(15, 20):
        gpus.append(GPU(str(i), '3080Ti'))
    return gpus


if __name__ == '__main__':
    time_type = 'predict'
    tasks = load_tasks(time_type)
    gpus = generate_gpus()
    gpus = SJF(tasks, gpus)
    for gpu in gpus:
        print(gpu.free_time)
    print(time_type)
    print(JCT(gpus))
    print(MAKESPAN(gpus))

    time_type = 'measure'
    tasks = load_tasks(time_type)
    gpus = generate_gpus()
    gpus = SJF(tasks, gpus)
    print(time_type)
    print(JCT(gpus))
    print(MAKESPAN(gpus))

    time_type = 'random'
    tasks = load_tasks(time_type)
    gpus = generate_gpus()
    gpus = SJF(tasks, gpus)
    print(time_type)
    print(JCT(gpus))
    print(MAKESPAN(gpus))
