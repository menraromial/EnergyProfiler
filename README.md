# EnergyProfiler

EnergyProfiler is a Python-based tool designed to measure and analyze the energy consumption of different types of computational workloads. It provides various functions that stress different aspects of system resources (CPU, memory, I/O) to help understand their energy impact.

## Features

- Multiple stress test functions targeting different system resources:
  - CPU-intensive operations (prime numbers, matrix calculations)
  - Memory-intensive operations (growing lists, matrices)
  - I/O operations (file reading/writing)
  - Network simulation
  - Mixed workloads
- Infinite execution mode with graceful interruption handling
- Command-line interface for easy function selection
- Execution time tracking

## Requirements

- Python 3.6+
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/menraromial/EnergyProfiler.git
cd EnergyProfiler
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the program from the command line, specifying the desired function:

```bash
python energy_profiler.py <function_name>
```

Available functions:
- `cpu_prime`: Prime number calculations
- `cpu_matrix`: Matrix operations
- `cpu_string`: String manipulations
- `cpu_trigo`: Trigonometric calculations
- `memory_list`: Growing list operations
- `memory_dict`: Dictionary operations
- `memory_matrix`: Matrix storage operations
- `mixed`: Mixed CPU and memory operations
- `io`: File I/O operations
- `network`: Network load simulation

Example:
```bash
python energy_profiler.py cpu_matrix
```

To stop the program, press Ctrl+C. The total execution time will be displayed.

## Project Structure

```
EnergyProfiler/
├── energy_profiler.py     # Main program file
├── requirements.txt       # Dependencies
└── README.md             # Documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.