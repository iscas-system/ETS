import json
from multiprocessing import Process
from django.http import JsonResponse, HttpRequest

from scheduler.scheduler import load_tasks, generate_gpus, SJF, JCT, MAKESPAN
from service.perf import do_perf, list_logs, do_predict, perf_detail

all_ps = []


def hello(request: HttpRequest) -> JsonResponse:
    data = {'message': 'Hello, World!'}
    return JsonResponse(data)


def perf_model(request: HttpRequest) -> JsonResponse:
    if request.method != 'POST':
        return JsonResponse({'message': 'method not allowed'}, status=405)

    model = request.POST.get('model', None)
    batch_size = request.POST.get('batch_size', None)
    input_size = request.POST.get('input_size', None)
    dtype = request.POST.get('dtype', None)
    gpu = request.POST.get('gpu', 'T4CPUALL')
    if model is None or batch_size is None or input_size is None or dtype is None:
        return JsonResponse({'message': 'bad request, parameter can not be null'}, status=400)
    # todo limit cpu
    p = Process(target=do_perf, args=(model, int(batch_size), int(input_size), dtype))
    p.start()

    return JsonResponse({'message': 'success'}, status=200)


def list_perf_result(request: HttpRequest) -> JsonResponse:
    res = list_logs()
    return JsonResponse({'data': res}, status=200)


def predict_perf_result(request: HttpRequest) -> JsonResponse:
    # gpu_name = request.GET.get('gpu_name', 'T4CPUALL')
    uuid = request.GET.get('uuid')
    gpu_name = 'T4CPUALL'
    if uuid is None:
        return JsonResponse({'message': 'bad request, parameter can not be null'}, status=400)
    result = do_predict(uuid, gpu_name)
    if result == 'success':
        return JsonResponse({'message': 'success'}, status=200)
    else:
        return JsonResponse({'message': result}, status=400)


def list_detail(request: HttpRequest) -> JsonResponse:
    uuid = request.GET.get('uuid')
    if uuid is None:
        return JsonResponse({'message': 'bad request, parameter can not be null'}, status=400)
    res = perf_detail(uuid)
    if isinstance(res, str):
        return JsonResponse({'message': res}, status=400)
    else:
        return JsonResponse({'data': res}, status=200)


def get_schedule_info(request: HttpRequest) -> JsonResponse:
    time_type = request.GET.get('type', 'predict')
    if time_type not in ['predict', 'measure', 'random']:
        return JsonResponse({'message': 'bad request, parameter can not be null'}, status=400)
    tasks = load_tasks(time_type)
    gpus = generate_gpus()
    gpus = SJF(tasks, gpus)

    return JsonResponse({'message': 'success',
                         'data': {
                             'schedule': [gpu.toJSON() for gpu in gpus],
                             'JCT': JCT(gpus),
                             'makespan': MAKESPAN(gpus)
                         }
                         }, status=200)
