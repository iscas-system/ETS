import torch

from torchvision import datasets, transforms


# CIFAR-100,
# mean, [0.5071, 0.4865, 0.4409]
# std, [0.2673, 0.2564, 0.2762]
def load_data(dataset_mode, batch_size, in_channel, h, w, dtype=torch.FloatTensor):
    input_data = torch.randn((batch_size, in_channel, h, w))
    if dtype == torch.FloatTensor:
        input_data = input_data.float()
    elif dtype == torch.HalfTensor:
        input_data = input_data.half()
    elif dtype == torch.DoubleTensor:
        input_data = input_data.double()

    if dataset_mode == "CIFAR10":
        output_data = torch.randn((batch_size, 10))
    elif dataset_mode == "CIFAR100":
        output_data = torch.randn((batch_size, 100))
    elif dataset_mode == "ImageNet":
        output_data = torch.randn((batch_size, 1000))
    elif dataset_mode == "MNIST":
        output_data = torch.randn((batch_size, 10))
    return input_data, output_data


def change_model_dtype(model, dtype):
    if dtype == torch.FloatTensor:
        return model.float()
    elif dtype == torch.HalfTensor:
        return model.half()
    else:
        return model.double()
# def load_data(dataset_mode, batch_size):
#     if dataset_mode is "CIFAR10":
#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ])
#
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ])
#         train_loader = torch.utils.data.DataLoader(
#             data_statistics.CIFAR10('data', train=True, download=True, transform=transform_train),
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=2
#         )
#
#         test_loader = torch.utils.data.DataLoader(
#             data_statistics.CIFAR10('data', train=False, transform=transform_test),
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=2
#         )
#     elif dataset_mode is "CIFAR100":
#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
#         ])
#
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
#         ])
#         train_loader = torch.utils.data.DataLoader(
#             data_statistics.CIFAR100('data', train=True, download=True, transform=transform_train),
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=2
#         )
#
#         test_loader = torch.utils.data.DataLoader(
#             data_statistics.CIFAR100('data', train=False, transform=transform_test),
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=2
#         )
#     elif dataset_mode is "MNIST":
#         transform_train = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,)),
#         ])
#
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,)),
#         ])
#         train_loader = torch.utils.data.DataLoader(
#             data_statistics.CIFAR100('data', train=True, download=True, transform=transform_train),
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=2
#         )
#
#         test_loader = torch.utils.data.DataLoader(
#             data_statistics.CIFAR100('data', train=False, transform=transform_test),
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=2
#         )
#     return train_loader, test_loader
#
#
