package simulation

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

const MatrixSize = 4
const NumCores = 2

func generateMatrix(size int) [][]float64 {
	matrix := make([][]float64, size)
	for i := range matrix {
		matrix[i] = make([]float64, size)
		for j := range matrix[i] {
			matrix[i][j] = rand.Float64()
		}
	}
	return matrix
}

func multiplyRow(row []float64, matrix [][]float64, result chan []float64, wg *sync.WaitGroup) {
	defer wg.Done()
	size := len(matrix)
	output := make([]float64, size)
	for j := 0; j < size; j++ {
		sum := 0.0
		for k := 0; k < size; k++ {
			sum += row[k] * matrix[k][j]
		}
		output[j] = sum
	}
	result <- output
}

func parallelMatrixMultiply(A, B [][]float64) [][]float64 {
	size := len(A)
	result := make([][]float64, size)
	resultChan := make(chan []float64, size)
	var wg sync.WaitGroup

	// Distribute rows across TPU cores
	for i := 0; i < size; i++ {
		wg.Add(1)
		go multiplyRow(A[i], B, resultChan, &wg)
	}

	// Collect results
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	for i := 0; i < size; i++ {
		result[i] = <-resultChan
	}

	return result
}

// RunMatrixMultiplication simulates parallel matrix multiplication
func RunMatrixMultiplication() {
	rand.Seed(time.Now().UnixNano())
	A := generateMatrix(MatrixSize)
	B := generateMatrix(MatrixSize)

	fmt.Println("Matrix A:", A)
	fmt.Println("Matrix B:", B)

	result := parallelMatrixMultiply(A, B)

	fmt.Println("Result:", result)
}
