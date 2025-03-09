package simulation

import (
	"fmt"
	"sync"
)

// Simulates an all-reduce operation across TPU cores
func allReduce(values []float64, numCores int) []float64 {
	size := len(values)
	partialSums := make([]float64, numCores)
	finalSum := make([]float64, size)
	var wg sync.WaitGroup

	// Step 1: Each TPU core computes a local sum
	for i := 0; i < numCores; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < size; j++ {
				partialSums[i] += values[j] / float64(numCores) // Average contribution
			}
		}(i)
	}

	wg.Wait()

	// Step 2: Aggregate across TPU cores
	for i := 0; i < size; i++ {
		for j := 0; j < numCores; j++ {
			finalSum[i] += partialSums[j]
		}
	}

	return finalSum
}

// RunAllReduce simulates all-reduce operation
func RunAllReduce() {
	data := []float64{1.0, 2.0, 3.0, 4.0}
	numCores := 4

	fmt.Println("Original Gradients:", data)
	reduced := allReduce(data, numCores)
	fmt.Println("All-Reduced Gradients:", reduced)
}
