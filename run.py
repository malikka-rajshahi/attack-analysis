import torch
import time
import argparse

def get_gpu_memory():
    return torch.cuda.get_device_properties(0).total_memory

def get_gpu_memory_usage():
    return torch.cuda.memory_allocated()

def continuous_matrix_multiply(duration, max_memory_usage=0.8):
    if not torch.cuda.is_available():
        #print("Error: GPU not available. This script requires a GPU.")
        return

    device = torch.device("cuda")
    total_memory = get_gpu_memory()
    max_memory = int(total_memory * max_memory_usage)
    
    start_time = time.time()
    matrix_size = 1000 
    iteration = 0
    
    while time.time() - start_time < duration:
        try:
            matrix1 = torch.rand(matrix_size, matrix_size, device=device)
            matrix2 = torch.rand(matrix_size, matrix_size, device=device)
            
            result = torch.matmul(matrix1, matrix2)
            
            del matrix1, matrix2, result
            if iteration % 2 == 0:
                torch.cuda.empty_cache()
            
            iteration += 1
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                matrix_size = int(matrix_size * 0.8)
                torch.cuda.empty_cache()
            else:
                raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-Aware Continuous GPU Matrix Multiplication")
    parser.add_argument("--max_memory", type=float, default=0.4, help="Maximum fraction of GPU memory to use (default: 0.4)")
    args = parser.parse_args()
    
    continuous_matrix_multiply(86400, args.max_memory)
