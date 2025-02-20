import argparse
import time
import numpy as np
from typing import List
import sys
import signal
import math
from datetime import datetime
import cmath
from functools import reduce

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

# Original functions remain the same...
def cpu_prime_numbers():
    """CPU intensive - Prime number calculation"""
    n = 2
    while True:
        #is_prime = True
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                _ = False
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
            _ = f.read()
        counter += 1

def network_simulation():
    """Network load simulation"""
    data = b"X" * 1000000
    while True:
        # Simulates network operations with computations
        _ = sum(data[i] * i for i in range(len(data)))
        time.sleep(0.1)

# New mathematical benchmark functions

def cpu_ackermann():
    """Ackermann function computation"""
    def ackermann(m: int, n: int) -> int:
        if m == 0:
            return n + 1
        elif n == 0:
            return ackermann(m - 1, 1)
        else:
            return ackermann(m - 1, ackermann(m, n - 1))
    
    while True:
        # Computing A(3,10) would overflow, so we use smaller values repeatedly
        for m in range(4):
            for n in range(4):
                ackermann(m, n)

def cpu_apery():
    """Calculate Apery's constant ζ(3) to high precision"""
    def apery_series(n: int) -> float:
        return 1.0 / (n ** 3)
    
    while True:
        result = 0.0
        # Sum to achieve precision of ~1.0e-14
        for n in range(1, 100000):
            result += apery_series(n)

def cpu_bitops():
    """Various bit operations benchmark"""
    def reverse_bits(n: int) -> int:
        result = 0
        for i in range(32):
            result = (result << 1) | (n & 1)
            n >>= 1
        return result
    
    def parity_check(n: int) -> int:
        parity = 0
        while n:
            parity ^= (n & 1)
            n >>= 1
        return parity
    
    def bit_count(n: int) -> int:
        count = 0
        while n:
            count += n & 1
            n >>= 1
        return count
    
    def nearest_power_of_2(n: int) -> int:
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        return n + 1
    
    while True:
        for i in range(1000000):
            reverse_bits(i)
            parity_check(i)
            bit_count(i)
            nearest_power_of_2(i)

def cpu_cfloat():
    """Complex floating point operations"""
    while True:
        z = complex(1.0, 1.0)
        for _ in range(1000):
            z = z * z.conjugate() + complex(0.5, 0.5)
            z = cmath.exp(z) / (z + complex(1.0, 1.0))
            z = cmath.sqrt(z * z.conjugate())

def cpu_double():
    """Double precision floating point operations"""
    while True:
        x = 1.0
        for _ in range(1000):
            x = math.sqrt(x * x + 1.0)
            x = math.log(x + 1.0)
            x = math.exp(x / 2.0)
            x = math.atan(x)

def cpu_euler():
    """Compute e using the limit definition"""
    def compute_e(n: int) -> float:
        return (1.0 + 1.0/n) ** n
    
    while True:
        for n in range(1000, 100000, 1000):
            compute_e(n)

def cpu_explog():
    """Iterate on exp(log(n))"""
    while True:
        x = 1.0
        for _ in range(10000):
            x = math.exp(math.log(x) / 1.00002)

def cpu_factorial():
    """Factorial approximations using Stirling and Ramanujan"""
    def stirling(n: int) -> float:
        return math.sqrt(2 * math.pi * n) * (n / math.e) ** n
    
    def ramanujan(n: int) -> float:
        return math.sqrt(math.pi) * (n / math.e) ** n * (8 * n ** 3 + 4 * n ** 2 + n + 1/30) ** (1/6)
    
    while True:
        for n in range(1, 151):
            stirling(n)
            ramanujan(n)

def cpu_fibonacci():
    """Compute Fibonacci sequence"""
    def fibonacci(n: int) -> int:
        if n <= 1:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)
    
    while True:
        for i in range(35):
            fibonacci(i)

def cpu_fft():
    """4096 sample Fast Fourier Transform"""
    while True:
        # Generate random signal
        signal = np.random.random(4096)
        # Compute FFT
        np.fft.fft(signal)

def cpu_gamma():
    """Calculate Euler-Mascheroni constant"""
    def harmonic_series(n: int) -> float:
        return sum(1.0/i for i in range(1, n + 1))
    
    while True:
        n = 80000
        gamma = harmonic_series(n) - math.log(n)

def cpu_gcd():
    """Compute Greatest Common Divisor"""
    def gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a
    
    while True:
        for a in range(1000):
            for b in range(1000):
                gcd(a, b)

def cpu_gray():
    """Binary to Gray code conversions"""
    def bin_to_gray(n: int) -> int:
        return n ^ (n >> 1)
    
    def gray_to_bin(n: int) -> int:
        mask = n
        while mask:
            mask >>= 1
            n ^= mask
        return n
    
    while True:
        for i in range(65536):
            gray = bin_to_gray(i)
            binary = gray_to_bin(gray)

def cpu_hamming():
    """Compute Hamming codes"""
    def compute_hamming(data: int) -> int:
        d1 = (data >> 0) & 1
        d2 = (data >> 1) & 1
        d3 = (data >> 2) & 1
        d4 = (data >> 3) & 1
        
        p1 = d2 ^ d3 ^ d4
        p2 = d1 ^ d3 ^ d4
        p3 = d1 ^ d2 ^ d4
        p4 = d1 ^ d2 ^ d3
        
        return (p1 << 7) | (p2 << 6) | (p3 << 5) | (p4 << 4) | data

    while True:
        for i in range(262144):
            compute_hamming(i & 0xF)

def cpu_hanoi():
    """Towers of Hanoi solver"""
    def hanoi(n: int, source: str, auxiliary: str, target: str) -> None:
        if n > 0:
            hanoi(n - 1, source, target, auxiliary)
            hanoi(n - 1, auxiliary, source, target)
    
    while True:
        hanoi(21, 'A', 'B', 'C')

def cpu_hyperbolic():
    """Hyperbolic function computations"""
    while True:
        for theta in np.linspace(0, 2*math.pi, 1500):
            sinh = math.sinh(theta)
            cosh = math.cosh(theta)
            result = sinh * cosh + math.sinh(2*theta) + math.cosh(3*theta)

def cpu_nsqrt():
    """Newton-Raphson square root computation"""
    def newton_sqrt(x: float, epsilon: float = 1e-10) -> float:
        if x < 0:
            raise ValueError("Square root of negative number")
        guess = x / 2
        while abs(guess * guess - x) > epsilon:
            guess = (guess + x / guess) / 2
        return guess
    
    while True:
        for x in range(1, 1000):
            newton_sqrt(float(x))

def cpu_omega():
    """Compute the omega constant"""
    def omega_iteration(omega: float) -> float:
        return (1 + omega) / (1 + math.exp(omega))
    
    while True:
        omega = 0.5  # Initial guess
        for _ in range(100):
            omega = omega_iteration(omega)

def cpu_phi():
    """Compute the Golden Ratio using series"""
    def compute_phi(iterations: int) -> float:
        phi = 1.0
        for _ in range(iterations):
            phi = 1 + 1/phi
        return phi
    
    while True:
        compute_phi(1000)

def cpu_pi():
    """Compute π using Ramanujan's algorithm"""
    def ramanujan_pi(terms: int) -> float:
        sum_series = 0
        for k in range(terms):
            num = math.factorial(4*k) * (1103 + 26390*k)
            den = (math.factorial(k)**4) * 396**(4*k)
            sum_series += num / den
        return 1 / ((2*math.sqrt(2)/9801) * sum_series)
    
    while True:
        ramanujan_pi(10)

def cpu_psi():
    """Compute the reciprocal Fibonacci constant"""
    def fibonacci_sequence(n: int) -> List[int]:
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib
    
    while True:
        fibs = fibonacci_sequence(100)
        _ = sum(1/x for x in fibs[1:])

def cpu_stats():
    """Statistical computations"""
    while True:
        # Generate random data
        data = np.random.random(250)
        
        # Calculate statistics
        _ = np.min(data)
        _ = np.max(data)
        _ = np.mean(data)
        _ = np.exp(np.mean(np.log(data)))
        _ = len(data) / np.sum(1/data)
        _ = np.std(data)

def cpu_trig():
    """Trigonometric computations"""
    while True:
        for theta in np.linspace(0, 2*math.pi, 1500):
            _ = (math.sin(theta) * math.cos(theta) + 
                     math.sin(2*theta) + math.cos(3*theta))

def cpu_zeta():
    """Compute Riemann Zeta function"""
    def zeta(s: float, terms: int = 1000) -> float:
        return sum(1/(n**s) for n in range(1, terms + 1))
    
    while True:
        for s in np.arange(2.0, 10.1, 0.1):
            zeta(s)

# Update run_infinite_benchmark function
def run_infinite_benchmark(func_name: str) -> None:
    """Runs the chosen function in an infinite loop"""
    functions = {
        # Original functions...
        'cpu_prime': cpu_prime_numbers,
        'cpu_matrix': cpu_matrix_operations,
        'cpu_string': cpu_string_operations,
        'cpu_trigo': cpu_trigonometry,
        'memory_list': memory_list_growth,
        'memory_dict': memory_dict_operations,
        'memory_matrix': memory_matrix_growth,
        'mixed': mixed_computation,
        'io': io_intensive,
        'network': network_simulation,
        
        # New mathematical functions
        'ackermann': cpu_ackermann,
        'apery': cpu_apery,
        'bitops': cpu_bitops,
        'cfloat': cpu_cfloat,
        'double': cpu_double,
        'euler': cpu_euler,
        'explog': cpu_explog,
        'factorial': cpu_factorial,
        'fibonacci': cpu_fibonacci,
        'fft': cpu_fft,
        'gamma': cpu_gamma,
        'gcd': cpu_gcd,
        'gray': cpu_gray,
        'hamming': cpu_hamming,
        'hanoi': cpu_hanoi,
        'hyperbolic': cpu_hyperbolic,
        'nsqrt': cpu_nsqrt,
        'omega': cpu_omega,
        'phi': cpu_phi,
        'pi': cpu_pi,
        'psi': cpu_psi,
        'stats': cpu_stats,
        'trig': cpu_trig,
        'zeta': cpu_zeta
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
                              'mixed', 'io', 'network',
                              'ackermann', 'apery', 'bitops', 'cfloat', 'double',
                              'euler', 'explog', 'factorial', 'fibonacci', 'fft',
                              'gamma', 'gcd', 'gray', 'hamming', 'hanoi',
                              'hyperbolic', 'nsqrt', 'omega', 'phi', 'pi',
                              'psi', 'stats', 'trig', 'zeta'],
                      help='Function to execute')
    
    args = parser.parse_args()
    run_infinite_benchmark(args.function)

if __name__ == "__main__":
    main()