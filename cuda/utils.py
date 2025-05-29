import time
from functools import wraps

from numba import cuda


def profile(runs=100):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            result = None

            for i in range(runs):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                times.append(end - start)

            avg_time = sum(times) / len(times)
            print(f"Function '{func.__name__}' executed {runs} time(s)")
            print(f"Average execution time: {avg_time:.4f} seconds")

            if runs > 1:
                min_time = min(times)
                max_time = max(times)
                print(f"Min: {min_time:.4f}s, Max: {max_time:.4f}s\n")

            return result

        return wrapper

    return decorator


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function '{func.__name__}' executed in {end - start:.4f} seconds")
        return result

    return wrapper


def get_cuda_device_info(verbose=True):
    """
    Retrieve comprehensive information about the current CUDA device.

    Parameters:
    -----------
    verbose : bool, optional
        If True, print device information. If False, return as dictionary.

    Returns:
    --------
    dict or None
        Dictionary of device properties if verbose=False, otherwise None
    """
    try:
        device = cuda.get_current_device()
        context = cuda.current_context()
        cuda_version = cuda.runtime.get_version()  # (major, minor)

        device_info = {
            "cuda_runtime_version": f"{cuda_version[0]}.{cuda_version[1]}",
            "name": device.name,
            "compute_capability": device.compute_capability,
            "total_memory_gb": context.get_memory_info().total / 1e9,
            "free_memory_gb": context.get_memory_info().free / 1e9,
            "multiprocessors": device.MULTIPROCESSOR_COUNT,
            "max_threads_per_block": device.MAX_THREADS_PER_BLOCK,
            "max_grid_dimensions": {
                "x": device.MAX_GRID_DIM_X,
                "y": device.MAX_GRID_DIM_Y,
                "z": device.MAX_GRID_DIM_Z,
            },
            "warp_size": device.WARP_SIZE,
            "clock_rate_khz": device.CLOCK_RATE,
            "memory_clock_rate_khz": device.MEMORY_CLOCK_RATE,
        }

        if verbose:
            print("CUDA Device Information:")
            print("-" * 40)
            print(
                f"{'CUDA Runtime Version:':<30} {device_info['cuda_runtime_version']}"
            )
            print(f"{'Device Name:':<30} {device_info['name']}")
            print(f"{'Compute Capability:':<30} {device_info['compute_capability']}")

            print("\nMemory:")
            print(f"{'Total Memory:':<30} {device_info['total_memory_gb']:.2f} GB")
            print(f"{'Free Memory:':<30} {device_info['free_memory_gb']:.2f} GB")

            print("\nCompute Resources:")
            print(
                f"{'Streaming Multiprocessors:':<30} {device_info['multiprocessors']}"
            )
            print(
                f"{'Max Threads per Block:':<30} {device_info['max_threads_per_block']}"
            )

            print("\nGrid Limitations:")
            print(
                f"{'Max Grid Dimensions X:':<30} {device_info['max_grid_dimensions']['x']}"
            )
            print(
                f"{'Max Grid Dimensions Y:':<30} {device_info['max_grid_dimensions']['y']}"
            )
            print(
                f"{'Max Grid Dimensions Z:':<30} {device_info['max_grid_dimensions']['z']}"
            )

            print("\nAdditional Characteristics:")
            print(f"{'Warp Size:':<30} {device_info['warp_size']}")
            print(f"{'Clock Rate:':<30} {device_info['clock_rate_khz']/1e6:.2f} GHz")
            print(
                f"{'Memory Clock Rate:':<30} {device_info['memory_clock_rate_khz']/1e6:.2f} GHz"
            )

        return device_info if not verbose else None

    except Exception as e:
        print(f"Error retrieving CUDA device information: {e}")
        return None
