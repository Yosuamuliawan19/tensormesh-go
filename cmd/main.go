package main

import (
	"fmt"
	"tpu-sim-go/pkg/simulation"
)

func main() {
	// Set up different simulations to run
	fmt.Println("Starting TPU Simulation...")

	// Run Matrix Multiplication Simulation
	fmt.Println("Running Parallel Matrix Multiplication Simulation...")
	simulation.RunMatrixMultiplication()

	// Run All-Reduce Simulation
	fmt.Println("Running All-Reduce Simulation...")
	simulation.RunAllReduce()
}
