package simulation

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

const NumTPUCores = 4 // Simulated TPU cores

// TPU Core structure
type TPUCore struct {
	ID      int
	InChan  chan []float64
	OutChan chan []float64
}

func NewTPUCore(id int) *TPUCore {
	return &TPUCore{
		ID:      id,
		InChan:  make(chan []float64, 1),
		OutChan: make(chan []float64, 1),
	}
}

// Simulates a TPU core processing data
func (t *TPUCore) Process(wg *sync.WaitGroup) {
	defer wg.Done()
	for data := range t.InChan {
		fmt.Printf("TPU Core %d processing data: %v\n", t.ID, data)
		// Simulate computation
		time.Sleep(time.Millisecond * time.Duration(rand.Intn(50)))
		// Send processed data to next TPU core
		t.OutChan <- data
	}
	close(t.OutChan) // Close when done
}

// RunSimulation simulates TPU core communication with multiple cores
func RunSimulation() {
	// Create TPU cores
	tpus := make([]*TPUCore, NumTPUCores)
	for i := 0; i < NumTPUCores; i++ {
		tpus[i] = NewTPUCore(i)
	}

	var wg sync.WaitGroup

	// Start TPU cores
	for _, tpu := range tpus {
		wg.Add(1)
		go tpu.Process(&wg)
	}

	// Send data to the first TPU core
	data := []float64{1.0, 2.0, 3.0}
	tpus[0].InChan <- data

	// Chain TPU cores (each one sends data to the next)
	for i := 0; i < NumTPUCores-1; i++ {
		go func(i int) {
			for result := range tpus[i].OutChan {
				tpus[i+1].InChan <- result
			}
			close(tpus[i+1].InChan) // Close when done
		}(i)
	}

	wg.Wait()
	fmt.Println("TPU Simulation Complete")
}
