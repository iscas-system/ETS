Profiling GPU workloads may require access to hardware performance
counters. Due to a fix in recent NVIDIA drivers addressing
CVE‑2018‑6260, the hardware counters are disabled by default, and
require elevated privileges to be enabled again. If you're using a recent
driver, you may see the following message when trying to run nvprof.

```bash
ERR_NVGPUCTRPERM The user running <tool_name/application_name> does not
have permission to access NVIDIA GPU Performance Counters on the target device.
```

For details, see [here](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters).

Permanent Solution
------------------

Follow the steps here. The current steps for Linux are:

```bash
sudo systemctl isolate multi-user
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia-vgpu-vfio nvidia
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
sudo systemctl isolate graphical
```

Temporary Solution
------------------

When running on bare metal, you can run nvprof with sudo.

If you're running in a Docker image, you can temporarily elevate your 
privileges with one of the following (oldest to newest syntax):

```bash
nvidia-docker run --privileged
docker run --runtime nvidia --privileged
docker run --gpus all --privileged
```
