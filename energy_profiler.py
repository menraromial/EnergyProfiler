import argparse
import time
import numpy as np
from typing import List, Dict
import sys
import signal
import math
from datetime import datetime

# Global variable to track execution time
start_time = None

def signal_handler(sig, frame):
    """Handler for Ctrl+C interruption"""
    if start_time is not None:
        duration = time.time() - start_time
        print(f"\nTotal execution time: {duration:.2f} seconds")
    print("\nStopping program")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def cpu_prime_numbers():
    """CPU intensive - Prime number calculation"""
    n = 2
    while True:
        is_prime = True
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                is_prime = False
                break
        n += 1

def cpu_matrix_operations():
    """CPU intensive - Matrix operations"""
    while True:
        size = 500
        matrix1 = np.random.rand(size, size)
        matrix2 = np.random.rand(size, size)
        result = np.dot(matrix1, matrix2)
        _ = np.linalg.eigvals(result)

def cpu_string_operations():
    """Moderate CPU - String manipulations"""
    text = "Hello" * 1000
    while True:
        text = text.upper()
        text = text.lower()
        text = text[::-1]
        text = "".join(sorted(text))

def cpu_trigonometry():
    """Moderate CPU - Trigonometric calculations"""
    x = 0.0
    while True:
        math.sin(x)
        math.cos(x)
        math.tan(x)
        x += 0.0001

def memory_list_growth():
    """Memory usage - Growing list"""
    data = []
    while True:
        data.extend([i for i in range(100000)])
        time.sleep(0.1)  # To avoid too rapid memory consumption

def memory_dict_operations():
    """Memory usage - Dictionary operations"""
    data = {}
    counter = 0
    while True:
        data[counter] = {i: np.random.rand(100) for i in range(100)}
        counter += 1
        if counter % 100 == 0:
            time.sleep(0.1)

def memory_matrix_growth():
    """Memory usage - Growing matrices"""
    matrices = []
    while True:
        matrices.append(np.random.rand(100, 100))
        time.sleep(0.1)

def mixed_computation():
    """Mixed CPU/Memory - Computation and storage"""
    data = []
    while True:
        matrix = np.random.rand(200, 200)
        result = np.linalg.svd(matrix)
        data.append(result)
        if len(data) > 100:
            data = data[-100:]

def io_intensive():
    """I/O intensive - File operations"""
    counter = 0
    while True:
        filename = f"temp_file_{counter % 5}.txt"
        with open(filename, 'w') as f:
            f.write(f"Timestamp: {datetime.now()}\n" * 1000)
        with open(filename, 'r') as f:
            content = f.read()
        counter += 1

def network_simulation():
    """Network load simulation"""
    data = b"X" * 1000000
    while True:
        # Simulates network operations with computations
        hash_value = sum(data[i] * i for i in range(len(data)))
        time.sleep(0.1)

def run_infinite_benchmark(func_name: str) -> None:
    """Runs the chosen function in an infinite loop"""
    functions = {
        'cpu_prime': cpu_prime_numbers,
        'cpu_matrix': cpu_matrix_operations,
        'cpu_string': cpu_string_operations,
        'cpu_trigo': cpu_trigonometry,
        'memory_list': memory_list_growth,
        'memory_dict': memory_dict_operations,
        'memory_matrix': memory_matrix_growth,
        'mixed': mixed_computation,
        'io': io_intensive,
        'network': network_simulation
    }
    
    if func_name not in functions:
        print(f"Function '{func_name}' not found")
        print(f"Available functions: {', '.join(functions.keys())}")
        sys.exit(1)
    
    print(f"Starting {func_name}")
    print("Press Ctrl+C to stop")
    
    global start_time
    start_time = time.time()
    
    functions[func_name]()

def main():
    parser = argparse.ArgumentParser(description='Energy consumption testing - Infinite mode')
    parser.add_argument('function', type=str, 
                      choices=['cpu_prime', 'cpu_matrix', 'cpu_string', 'cpu_trigo',
                              'memory_list', 'memory_dict', 'memory_matrix',
                              'mixed', 'io', 'network'],
                      help='Function to execute')
    
    args = parser.parse_args()
    run_infinite_benchmark(args.function)

if __name__ == "__main__":
    main()