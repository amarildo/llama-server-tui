import ctypes
import sys

def sweep_vram():
    try:
        # Load CUDA runtime dynamically
        cudart = ctypes.CDLL('libcudart.so')
    except OSError:
        print("CUDA runtime not found. Are you sure CUDA is installed?")
        return

    # Check for available devices
    count = ctypes.c_int()
    cudart.cudaGetDeviceCount(ctypes.byref(count))
    if count.value == 0:
        print("No CUDA devices found.")
        return
        
    # Get free and total memory
    free_mem = ctypes.c_size_t()
    total_mem = ctypes.c_size_t()
    cudart.cudaMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem))
    
    # Allocate 95% of free memory to force garbage collection/defragmentation
    alloc_size = int(free_mem.value * 0.95)
    print(f"Sweeping {alloc_size / (1024**3):.2f} GB of VRAM to defragment memory...")
    
    ptr = ctypes.c_void_p()
    # Try allocating
    res = cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(alloc_size))
    
    if res == 0: # cudaSuccess
        # Write 0s to the memory to ensure physical pages are mapped by the WSL driver
        cudart.cudaMemset(ptr, 0, ctypes.c_size_t(alloc_size))
        
        # Free the memory back to the system
        cudart.cudaFree(ptr)
        print("Success! VRAM has been defragmented. You can now launch llama-server.")
    else:
        print(f"Failed to sweep memory (CUDA error {res}). Try running it again.")

if __name__ == "__main__":
    sweep_vram()
