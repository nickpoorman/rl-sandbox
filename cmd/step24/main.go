/**
* Adding a Learning Rate Scheduler
**/

// Up until now, we've used a constant learning rate or adapted it based on the
// episode number. However, more sophisticated schedules can help in learning
// more effectively. A learning rate scheduler adjusts the learning rate during
// training, typically reducing it as training progresses. This can help to
// fine-tune the agent's policy as it gets closer to an optimal policy.

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
	Gamma            = 0.9
	ReplayBufferSize = 1000
	BatchSize        = 32
	BaseLearningRate = 0.01
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
	Policy       []int
	Model        *NeuralNet
	TargetModel  *NeuralNet
	Epsilon      float64
	Visits       [][]int
	TargetUpdate int // How many steps to take before updating the target network
	StepCount    int // Keep track of the number of steps taken
	LearningRate float64
}

func NewAgent(numStates, numActions int, epsilon float64, targetUpdate int) *Agent {
	policy := make([]int, numStates)
	model := NewNeuralNet(numStates, numActions)
	targetModel := NewNeuralNet(numStates, numActions)
	copy(targetModel.Weights, model.Weights)
	copy(targetModel.Biases, model.Biases)
	visits := make([][]int, numStates)
	for s := 0; s < numStates; s++ {
		visits[s] = make([]int, numActions)
	}
	return &Agent{
		Policy:       policy,
		Model:        model,
		TargetModel:  targetModel,
		Epsilon:      epsilon,
		Visits:       visits,
		TargetUpdate: targetUpdate,
	}
}

func (agent *Agent) UpdateLearningRate(step int) {
	agent.LearningRate = learningRateScheduler(step)
}

// Here, we’ve replaced the epsilon-greedy strategy with softmax exploration,
// which selects actions based on their estimated values, creating a probability
// distribution where better actions are more likely to be selected. This
// approach encourages exploration of not just random actions but also actions
// that the agent believes are promising.
func (agent *Agent) Act(state int) int {
	input := make([]float64, agent.Model.NumInput)
	input[state] = 1
	qValues := agent.Model.Forward(input)

	var sumExpQ float64
	expQ := make([]float64, NumActions)
	for a := 0; a < NumActions; a++ {
		expQ[a] = math.Exp(qValues[a])
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
	input := make([]float64, agent.Model.NumInput)
	input[state] = 1

	targetOutput := agent.Model.Forward(input)
	targetOutput[action] = reward

	if newState != -1 {
		nextInput := make([]float64, agent.TargetModel.NumInput)
		nextInput[newState] = 1
		nextQValues := agent.TargetModel.Forward(nextInput)

		// Assuming a discount factor (Gamma) of 0.9
		targetOutput[action] += Gamma * max(nextQValues)
	}

	agent.Model.Update(input, targetOutput, agent.LearningRate) // learningRate of 0.01

	updatedQValues := agent.Model.Forward(input)
	if updatedQValues[0] > updatedQValues[1] {
		agent.Policy[state] = 0
	} else {
		agent.Policy[state] = 1
	}

	// update the target network every TargetUpdate steps.
	agent.StepCount++
	if agent.StepCount%agent.TargetUpdate == 0 {
		copy(agent.TargetModel.Weights, agent.Model.Weights)
		copy(agent.TargetModel.Biases, agent.Model.Biases)
	}
}

func max(slice []float64) float64 {
	maxValue := slice[0]
	for _, v := range slice {
		if v > maxValue {
			maxValue = v
		}
	}
	return maxValue
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

// Neural network
type NeuralNet struct {
	Weights   []float64
	Biases    []float64
	NumInput  int
	NumOutput int
}

func NewNeuralNet(numInput, numOutput int) *NeuralNet {
	weights := make([]float64, numInput*numOutput)
	biases := make([]float64, numOutput)
	return &NeuralNet{Weights: weights, Biases: biases, NumInput: numInput, NumOutput: numOutput}
}

func (nn *NeuralNet) Forward(inputs []float64) []float64 {
	outputs := make([]float64, nn.NumOutput)

	for o := 0; o < nn.NumOutput; o++ {
		for i := 0; i < nn.NumInput; i++ {
			outputs[o] += inputs[i] * nn.Weights[o*nn.NumInput+i]
		}
		outputs[o] += nn.Biases[o]
		outputs[o] = math.Tanh(outputs[o]) // Activation function
	}

	return outputs
}

func (nn *NeuralNet) Update(inputs []float64, targetOutputs []float64, learningRate float64) {
	predictions := nn.Forward(inputs)

	for o := 0; o < nn.NumOutput; o++ {
		err := targetOutputs[o] - predictions[o]
		for i := 0; i < nn.NumInput; i++ {
			nn.Weights[o*nn.NumInput+i] += learningRate * err * inputs[i]
		}
		nn.Biases[o] += learningRate * err
	}
}

// The learning rate decreases exponentially with the number of steps. The base
// rate is set to 0.01, and it decreases as the step number increases.
func learningRateScheduler(step int) float64 {
	return BaseLearningRate * math.Exp(-float64(step)/1000.0)
}

func main() {
	totalEpisodeRewards := 0.0
	maxEpisodeReward := 0.0

	for episode := 0; episode < 1000; episode++ {
		env := Environment{}                                                         // reset environment at the start of each episode
		agent := NewAgent(NumStates, NumActions, 1.0/(1.0+float64(episode)/100), 50) // decay epsilon
		replayBuffer := NewReplayBuffer(ReplayBufferSize)

		totalReward := 0.0
		for step := 0; step < 100; step++ {
			maxEpisodeReward += 0.97
			agent.UpdateLearningRate(agent.StepCount)

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

		totalEpisodeRewards += totalReward
		fmt.Printf("Episode %d finished. Total Reward: %.2f\n", episode+1, totalReward)
	}

	fmt.Printf("Episodes finished. Total Episode Reward: %.2f / %.2f\n", totalEpisodeRewards, maxEpisodeReward)
}

// By introducing a learning rate scheduler, we've added a mechanism to adapt
// the learning rate as the agent gains experience. This can lead to better
// convergence and performance, particularly in complex environments. The
// learning rate scheduler ensures that the agent makes large updates when it is
// far from the optimal policy and smaller, more refined updates as it gets
// closer, leading to a more stable and reliable learning process.
