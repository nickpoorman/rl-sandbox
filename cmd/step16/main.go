/**
* Function Approximation with a Linear Model
**/

// We'll replace our Q-value table with a linear function approximation. For
// simplicity, we will use a linear model to approximate the Q-values.

package main

import (
	"fmt"
	"math"
	"math/rand"
)

type Action int

const (
	MoveLeft Action = iota
	MoveRight
)

const (
	NumStates        = 5
	NumActions       = 2 // 0: move left, 1: move right
	Epsilon          = 0.1
	Gamma            = 0.99
	ReplayBufferSize = 1000
	BatchSize        = 32
)

// Environment struct to represent our world
type Environment struct {
	State int
}

// Step function to take an action and move the agent
func (env *Environment) Step(action int) (int, float64, bool) {
	oldState := env.State
	if action == 0 && env.State > 0 {
		env.State--
	} else if action == 1 && env.State < NumStates-1 {
		env.State++
	}

	reward := -0.01 // small negative reward for each step
	if env.State == NumStates-1 && oldState != env.State {
		reward = 1.0
	}

	done := env.State == NumStates-1

	return env.State, reward, done
}

// Agent struct to represent our learning agent
type Agent struct {
	Policy  []int
	Model   *LinearModel
	Epsilon float64
	Visits  [][]int
}

func NewAgent(numStates, numActions int, epsilon float64) *Agent {
	policy := make([]int, numStates)
	model := NewLinearModel(numStates, numActions)
	visits := make([][]int, numStates)
	for s := 0; s < numStates; s++ {
		visits[s] = make([]int, numActions)
	}
	return &Agent{Policy: policy, Model: model, Epsilon: epsilon, Visits: visits}
}

// Here, we’ve replaced the epsilon-greedy strategy with softmax exploration,
// which selects actions based on their estimated values, creating a probability
// distribution where better actions are more likely to be selected. This
// approach encourages exploration of not just random actions but also actions
// that the agent believes are promising.
func (agent *Agent) Act(state int) int {
	var sumExpQ float64
	expQ := make([]float64, NumActions)
	for a := 0; a < NumActions; a++ {
		expQ[a] = math.Exp(agent.Model.Predict(state, a))
		sumExpQ += expQ[a]
	}

	prob := make([]float64, NumActions)
	for a := 0; a < NumActions; a++ {
		prob[a] = expQ[a] / sumExpQ
	}

	randVal := rand.Float64()
	sumProb := 0.0
	for a := 0; a < NumActions; a++ {
		sumProb += prob[a]
		if randVal < sumProb {
			return a
		}
	}
	return NumActions - 1
}

func (agent *Agent) UpdateQValues(state, action int, reward float64, newState int) {
	learningRate := 1.0 / (1.0 + float64(agent.Visits[state][action]))
	agent.Visits[state][action]++

	oldValue := agent.Model.Predict(state, action)
	rewardPlusValue := reward + Gamma*agent.Model.Predict(newState, agent.Policy[newState])
	agent.Model.Update(state, action, oldValue+learningRate*(rewardPlusValue-oldValue), learningRate)

	// Update the policy to take the best action from this state
	if agent.Model.Predict(state, 0) > agent.Model.Predict(state, 1) {
		agent.Policy[state] = 0
	} else {
		agent.Policy[state] = 1
	}
}

type Experience struct {
	State     int
	Action    int
	Reward    float64
	NextState int
}

type ReplayBuffer struct {
	Buffer  []Experience
	Counter int
	Size    int
}

func NewReplayBuffer(size int) *ReplayBuffer {
	return &ReplayBuffer{
		Buffer: make([]Experience, size),
		Size:   size,
	}
}

func (rb *ReplayBuffer) Store(e Experience) {
	rb.Buffer[rb.Counter%rb.Size] = e
	rb.Counter++
}

func (rb *ReplayBuffer) Sample(batchSize int) []Experience {
	samples := make([]Experience, batchSize)
	for i := 0; i < batchSize; i++ {
		idx := rand.Intn(min(rb.Counter, rb.Size))
		samples[i] = rb.Buffer[idx]
	}
	return samples
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type LinearModel struct {
	// represents the weights of our linear function
	Weights [][]float64
}

func NewLinearModel(numStates, numActions int) *LinearModel {
	weights := make([][]float64, numStates)
	for s := 0; s < numStates; s++ {
		weights[s] = make([]float64, numActions)
	}
	return &LinearModel{Weights: weights}
}

// Predict computes the Q-value for a given state-action pair
func (lm *LinearModel) Predict(state, action int) float64 {
	return lm.Weights[state][action]
}

// Update performs a gradient update to adjust the weights based on the target Q-value
func (lm *LinearModel) Update(state, action int, target float64, learningRate float64) {
	prediction := lm.Predict(state, action)
	err := target - prediction
	lm.Weights[state][action] += learningRate * err
}

func main() {
	for episode := 0; episode < 10000; episode++ {
		env := Environment{}                                                     // reset environment at the start of each episode
		agent := NewAgent(NumStates, NumActions, 1.0/(1.0+float64(episode)/100)) // decay epsilon
		replayBuffer := NewReplayBuffer(ReplayBufferSize)

		totalReward := 0.0
		for step := 0; step < 100; step++ {
			state := env.State
			action := agent.Act(state)
			newState, reward, done := env.Step(action)
			agent.UpdateQValues(state, action, reward, newState)
			totalReward += reward

			fmt.Printf("Episode %d, Step %d: State: %d, Action: %d, New State: %d, Reward: %.2f\n", episode+1, step+1, state, action, newState, reward)

			// Store experience in replay buffer
			experience := Experience{State: state, Action: action, Reward: reward, NextState: newState}
			replayBuffer.Store(experience)

			// Sample a batch from replay buffer and update Q-values
			if replayBuffer.Counter > BatchSize {
				batch := replayBuffer.Sample(BatchSize)
				for _, e := range batch {
					agent.UpdateQValues(e.State, e.Action, e.Reward, e.NextState)
				}
			}

			if done {
				fmt.Printf("Episode %d finished after %d steps. Total Reward: %.2f\n", episode+1, step+1, totalReward)
				break
			}
		}

		fmt.Printf("Episode %d finished. Total Reward: %.2f\n", episode+1, totalReward)
	}

}
