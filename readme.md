# Distributed TPU tensor calculation

This project is a **Go** implementation of a **TPU communication simulator**. It includes simulations for various aspects of TPU computations, such as **core processing**, **parallel matrix multiplication**, and **all-reduce communication**.

This project is primarily for self-study and personal exploration. Its a proof of concept implementation on distributed TPU architectures and concepts in distributed ML acceleration.

## Project Structure

- **cmd/main.go**: Entry point for running different simulations.
- **pkg/simulation/**: Contains different simulation logic files, including:
  - `tpu_sim.go`: Simulates TPU core processing and communication.
  - `matrix.go`: Simulates parallel matrix multiplication across TPU cores.
  - `allreduce.go`: Simulates all-reduce operation for gradient aggregation.

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/tpu-sim-go.git
   cd tpu-sim-go
   ```

2. Run the simulations:
   ```bash
   go run cmd/main.go
   ```

## Simulations Available

- **TPU Core Simulation**: Simulates multiple TPU cores processing and communicating data in parallel.
- **Parallel Matrix Multiplication**: Simulates matrix multiplication across multiple TPU cores.
- **All-Reduce**: Simulates the all-reduce operation, typically used for gradient aggregation in distributed training.

## Requirements

- Go 1.18+ installed on your machine.

## Contributing

Feel free to contribute! Create an issue or submit a pull request for new simulations or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
