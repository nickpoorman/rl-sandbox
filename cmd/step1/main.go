package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

type Metrics struct {
	timestamp     int64
	windowSize    int
	latency       int
	unackMessages int
}

type State struct {
	windowSize    int
	latency       int
	unackMessages int
}

type Action int

const (
	Increase Action = iota
	Decrease
	KeepSame
)

type QValue struct {
	state  State
	action Action
	value  float64
}

var LearningRate = 0.1
var DiscountFactor = 0.9
var Epsilon = 0.2

var mu sync.Mutex
var QTable = make(map[State]map[Action]float64)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func getQValue(s State, a Action) float64 {
	if _, ok := QTable[s]; !ok {
		QTable[s] = make(map[Action]float64)
	}
	return QTable[s][a]
}

func setQValue(s State, a Action, v float64) {
	if _, ok := QTable[s]; !ok {
		QTable[s] = make(map[Action]float64)
	}
	QTable[s][a] = v
}

func chooseAction(s State) Action {
	if rand.Float64() < Epsilon {
		return Action(rand.Intn(3))
	}

	maxVal := -1e9
	var bestAction Action
	mu.Lock()
	for a := Increase; a <= KeepSame; a++ {
		if getQValue(s, a) > maxVal {
			maxVal = getQValue(s, a)
			bestAction = a
		}
	}
	mu.Unlock()
	return bestAction
}

func main() {
	LearningRate, DiscountFactor, Epsilon = hyperparameterTuning()

	c := make(chan Metrics)
	stop := make(chan struct{})

	// Start the continuous training goroutine
	go continuousTraining(stop, c)

	// Mock ingestion of metrics
	go func() {
		for i := 0; i < 250000; i++ {
			metrics := Metrics{
				timestamp:     time.Now().Unix(),
				windowSize:    rand.Intn(10000) + 1,
				latency:       rand.Intn(500),
				unackMessages: rand.Intn(10000),
			}
			select {
			case c <- metrics:
			case <-stop:
			}

			// time.Sleep(10 * time.Second)
		}

		func() { // print q values
			mu.Lock() // make sure to lock for reading
			defer mu.Unlock()

			for stateAction, qValue := range QTable {
				fmt.Printf("State-Action: %s, Q-Value: %f\n", stateAction, qValue)
			}
		}()
	}()

	// Allow predictions anytime
	for {
		fmt.Println("Enter windowSize, latency, unackMessages (comma separated):")
		var ws, lat, unack int
		_, err := fmt.Scanf("%d,%d,%d", &ws, &lat, &unack)
		if err != nil {
			fmt.Println("Invalid input. Try again.")
			continue
		}

		currentState := State{
			windowSize:    ws,
			latency:       lat,
			unackMessages: unack,
		}
		action := predictOptimalAction(currentState)
		fmt.Printf("Optimal action for state %+v is %d\n", currentState, action)
	}
}

func bestActionForState(s State) Action {
	maxVal := math.SmallestNonzeroFloat64
	var bestAction Action
	for a := Increase; a <= KeepSame; a++ {
		if getQValue(s, a) > maxVal {
			maxVal = getQValue(s, a)
			bestAction = a
		}
	}
	return bestAction
}

func continuousTraining(stop chan struct{}, c chan Metrics) {
	metricsProcessed := 0
	const PrintFrequency = 10 // Change this to set how often you want to print
	var avgQValueChange float64

	for {
		select {
		case <-stop:
			return
		case metrics := <-c:
			metricsProcessed++

			currentState := State{
				windowSize:    metrics.windowSize,
				latency:       metrics.latency,
				unackMessages: metrics.unackMessages,
			}

			for step := 0; step < 100; step++ {
				action := chooseAction(currentState)
				nextState, reward := simulateEnvironment(currentState, action)

				// Q-learning formula
				mu.Lock()
				oldValue := getQValue(currentState, action)
				nextMax := getQValue(nextState, Increase)
				for a := Increase; a <= KeepSame; a++ {
					if getQValue(nextState, a) > nextMax {
						nextMax = getQValue(nextState, a)
					}
				}
				mu.Unlock()

				newValue := oldValue + LearningRate*(reward+DiscountFactor*nextMax-oldValue)
				avgQValueChange += math.Abs(newValue - oldValue)

				mu.Lock()
				setQValue(currentState, action, newValue)
				mu.Unlock()

				currentState = nextState
			}

			// "Forgetting" mechanism: occasionally reset some Q-values to simulate adapting to changes
			if rand.Float64() < 0.05 {
				randomState := State{
					windowSize:    rand.Intn(10000) + 1,
					latency:       rand.Intn(500),
					unackMessages: rand.Intn(10000),
				}
				mu.Lock()
				delete(QTable, randomState)
				mu.Unlock()
			}

			// Periodically print out the number of states and average change in Q-values
			if metricsProcessed%PrintFrequency == 0 {
				mu.Lock()
				fmt.Printf("Processed %d metrics. QTable size: %d. Avg Q-value change: %f\n", metricsProcessed, len(QTable), avgQValueChange/float64(PrintFrequency*100))
				mu.Unlock()
				avgQValueChange = 0
			}
		}
	}
}

func predictOptimalAction(s State) Action {
	mu.Lock()
	defer mu.Unlock()
	return bestActionForState(s)
}

// A larger window size is generally desired, so we'll give a reward
// proportional to the window size. However, there are trade-offs. A larger
// window size that results in high latency or a large number of unacknowledged
// messages is not desirable. Thus, we'll penalize these conditions.
func getReward(state State, action Action) float64 {
	// Base reward is proportional to the window size
	reward := float64(state.windowSize)

	// Penalties for undesirable conditions
	if state.latency > 3 {
		reward -= float64(state.latency * 2)
	}
	if state.unackMessages > 5 {
		reward -= float64(state.unackMessages * 3)
	}

	return reward
}

// The main idea is to make the environment such that increasing the windowSize
// may come with some costs (like increased latency, unacknowledged messages,
// etc.), but the agent should learn to balance these trade-offs and optimize
// for a larger windowSize without incurring too many penalties.
func simulateEnvironment(s State, a Action) (State, float64) {
	newState := s

	switch a {
	case Increase:
		newState.windowSize++
		// As window size increases, there's a chance latency, unackMessages, and inFlightMessages increase
		if rand.Float32() < 0.4 {
			newState.latency++
		}
		if rand.Float32() < 0.5 {
			newState.unackMessages++
		}
	case Decrease:
		newState.windowSize--
		// As window size decreases, there's a chance latency, unackMessages, and inFlightMessages decrease
		if rand.Float32() < 0.4 {
			newState.latency--
		}
		if rand.Float32() < 0.5 {
			newState.unackMessages--
		}
	}

	// Ensure no negative values
	if newState.windowSize < 1 {
		newState.windowSize = 1
	}
	if newState.latency < 0 {
		newState.latency = 0
	}
	if newState.unackMessages < 0 {
		newState.unackMessages = 0
	}

	if newState.windowSize > 10000 {
		newState.windowSize = 10000
	}
	if newState.latency > 10000 {
		newState.latency = 10000
	}
	if newState.unackMessages > 10000 {
		newState.unackMessages = 10000
	}

	reward := getReward(newState, a)
	return newState, reward
}

func hyperparameterTuning() (float64, float64, float64) {
	learningRates := []float64{0.01, 0.1, 0.2}
	discountFactors := []float64{0.9, 0.95, 0.99}
	epsilons := []float64{0.1, 0.2, 0.3}

	bestParams := struct {
		LearningRate   float64
		DiscountFactor float64
		Epsilon        float64
		Performance    float64
	}{
		Performance: math.Inf(-1),
	}

	for _, lr := range learningRates {
		for _, df := range discountFactors {
			for _, e := range epsilons {
				// Set the parameters
				LearningRate = lr
				DiscountFactor = df
				Epsilon = e

				// Reset the Q-table and any other necessary state
				mu.Lock()
				QTable = make(map[State]map[Action]float64)
				mu.Unlock()

				// Run your simulation
				performance := runSimulation() // Implement this function to run your training and return a performance metric

				// Update best parameters if necessary
				if performance > bestParams.Performance {
					bestParams.LearningRate = lr
					bestParams.DiscountFactor = df
					bestParams.Epsilon = e
					bestParams.Performance = performance
				}
			}
		}
	}

	fmt.Printf("Best Parameters: LearningRate: %f, DiscountFactor: %f, Epsilon: %f, Performance: %f\n",
		bestParams.LearningRate, bestParams.DiscountFactor, bestParams.Epsilon, bestParams.Performance)

	return bestParams.LearningRate, bestParams.DiscountFactor, bestParams.Epsilon
}

func runSimulation() float64 {
	// Number of episodes for training
	numEpisodes := 1000

	// Channel for metrics
	c := make(chan Metrics)
	stop := make(chan struct{})

	// Start the continuous training goroutine
	go continuousTraining(stop, c)

	// Mock ingestion of metrics for training
	for episode := 0; episode < numEpisodes; episode++ {
		metrics := Metrics{
			timestamp:     time.Now().Unix(),
			windowSize:    rand.Intn(10000) + 1,
			latency:       rand.Intn(500),
			unackMessages: rand.Intn(10000),
		}
		c <- metrics
	}

	close(stop)

	// Evaluate performance after training
	performance := evaluatePerformance()

	return performance
}

func evaluatePerformance() float64 {
	// Number of test episodes
	numTestEpisodes := 1000
	cumulativeReward := 0.0

	// Run the model through a series of test episodes
	for episode := 0; episode < numTestEpisodes; episode++ {
		// Initial random state
		currentState := State{
			windowSize:    rand.Intn(10000) + 1,
			latency:       rand.Intn(500),
			unackMessages: rand.Intn(10000),
		}

		// Run through a series of actions based on the model's policy
		for step := 0; step < 100; step++ {
			action := predictOptimalAction(currentState)
			nextState, reward := simulateEnvironment(currentState, action)
			cumulativeReward += reward
			currentState = nextState
		}
	}

	// Return the average reward per episode
	return cumulativeReward / float64(numTestEpisodes)
}
