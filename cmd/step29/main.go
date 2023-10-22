/**
* Reward Normalization
**/

// Policy evaluation is an important aspect of reinforcement learning, as it
// allows us to estimate the value function of a policy, which is essential for
// understanding how good a given policy is.

// Policy evaluation is the process of assessing a policy to determine its
// effectiveness. This is done by calculating the value function of the policy,
// which gives us the expected cumulative reward that can be obtained from each
// state while following the policy.
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
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

	// Reward Normalization Parameters
	RewardMean float64
	RewardStd  float64
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

		// Reward Normalization Parameters
		RewardMean: 0.0,
		RewardStd:  1.0, // Initialize to 1.0 prevent division by zero
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

func (agent *Agent) UpdateQValues(state, action int, reward float64, newState int, done bool) {
	// Normalizing the reward
	normalizedReward := (reward - agent.RewardMean) / agent.RewardStd

	input := make([]float64, agent.Model.NumInput)
	input[state] = 1

	targetOutput := agent.Model.Forward(input)
	targetOutput[action] = normalizedReward

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

	// Update the running mean and standard deviation of the rewards
	if !done {
		agent.RewardMean = 0.99*agent.RewardMean + 0.01*reward
		rewardDiff := reward - agent.RewardMean
		agent.RewardStd = math.Sqrt(0.99*agent.RewardStd*agent.RewardStd + 0.01*rewardDiff*rewardDiff)
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
	return &NeuralNet{
		Weights:   weights,
		Biases:    biases,
		NumInput:  numInput,
		NumOutput: numOutput,
	}
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

func (nn *NeuralNet) Backward(inputs []float64, gradOutputs []float64, learningRate float64) {
	// Calculate the gradient of the loss with respect to the inputs
	gradInputs := make([]float64, nn.NumInput)
	for i := 0; i < nn.NumInput; i++ {
		for o := 0; o < nn.NumOutput; o++ {
			// Derivative of tanh(x) is 1 - tanh^2(x)
			outputBeforeActivation := nn.Forward(inputs)[o]
			derivTanh := 1 - math.Pow(math.Tanh(outputBeforeActivation), 2)
			gradInputs[i] += gradOutputs[o] * derivTanh * nn.Weights[o*nn.NumInput+i]
		}
	}

	// Update weights and biases
	for o := 0; o < nn.NumOutput; o++ {
		for i := 0; i < nn.NumInput; i++ {
			// Derivative of tanh(x) is 1 - tanh^2(x)
			outputBeforeActivation := nn.Forward(inputs)[o]
			derivTanh := 1 - math.Pow(math.Tanh(outputBeforeActivation), 2)
			gradWeight := gradOutputs[o] * derivTanh * inputs[i]
			nn.Weights[o*nn.NumInput+i] -= learningRate * gradWeight
		}
		gradBias := gradOutputs[o] * (1 - math.Pow(nn.Forward(inputs)[o], 2))
		nn.Biases[o] -= learningRate * gradBias
	}
}

func (nn *NeuralNet) Update(input []float64, target []float64, learningRate float64) {
	// Forward pass
	output := nn.Forward(input)

	// Compute the gradient of the loss with respect to the output
	gradOutput := make([]float64, len(output))
	for i := range output {
		gradOutput[i] = 2 * (output[i] - target[i])
	}

	// Clip gradients to be within [-1, 1]
	for i := range gradOutput {
		if gradOutput[i] > 1.0 {
			gradOutput[i] = 1.0
		} else if gradOutput[i] < -1.0 {
			gradOutput[i] = -1.0
		}
	}

	// Backward pass and update weights
	nn.Backward(input, gradOutput, learningRate)
}

// The learning rate decreases exponentially with the number of steps. The base
// rate is set to 0.01, and it decreases as the step number increases.
func learningRateScheduler(step int) float64 {
	return BaseLearningRate * math.Exp(-float64(step)/1000.0)
}

func sum(vals []float64) float64 {
	var total float64
	for _, v := range vals {
		total += v
	}
	return total
}

func visualizeLearningProgress(totalRewards []float64) {
	p := plot.New()

	p.Title.Text = "Learning Progress"
	p.X.Label.Text = "Episodes"
	p.Y.Label.Text = "Total Reward"

	pts := make(plotter.XYs, len(totalRewards))
	for i := range totalRewards {
		pts[i].X = float64(i)
		pts[i].Y = totalRewards[i]
	}

	line, err := plotter.NewLine(pts)
	if err != nil {
		log.Fatalf("Could not create line plotter: %v", err)
	}

	p.Add(line)
	p.Legend.Add("Total Reward", line)

	if err := p.Save(6*vg.Inch, 6*vg.Inch, "learning_progress.png"); err != nil {
		log.Fatalf("Could not save plot to file: %v", err)
	}

	fmt.Println("Learning progress plot saved to learning_progress.png")
}

func main() {
	numEpisodes := 1000
	maxPossibleTotalRewards := 0.0
	totalRewards := make([]float64, numEpisodes)

	var (
		// Early Stopping Parameters
		minImprovement float64 = 0.01             // Minimum improvement to consider as progress
		patience       int     = 10               // Number of episodes to wait for improvement before stopping
		waitCounter    int     = 0                // Counter to keep track of episodes without improvement
		bestReward     float64 = -math.MaxFloat64 // Initialize with the lowest possible value
	)

	for episode := 0; episode < numEpisodes; episode++ {
		env := Environment{}                                                         // reset environment at the start of each episode
		agent := NewAgent(NumStates, NumActions, 1.0/(1.0+float64(episode)/100), 50) // decay epsilon
		replayBuffer := NewReplayBuffer(ReplayBufferSize)

		totalReward := 0.0
		for step := 0; step < 1000; step++ {
			maxPossibleTotalRewards += 0.97
			agent.UpdateLearningRate(agent.StepCount)

			state := env.State
			action := agent.Act(state)
			newState, reward, done := env.Step(action)
			agent.UpdateQValues(state, action, reward, newState, done)
			totalReward += reward

			fmt.Printf("Episode %d, Step %d: State: %d, Action: %d, New State: %d, Reward: %.2f\n", episode+1, step+1, state, action, newState, reward)

			// Store experience in replay buffer
			experience := Experience{State: state, Action: action, Reward: reward, NextState: newState}
			replayBuffer.Store(experience)

			// Sample a batch from replay buffer and update Q-values
			if replayBuffer.Counter > BatchSize {
				batch := replayBuffer.Sample(BatchSize)
				for _, e := range batch {
					agent.UpdateQValues(e.State, e.Action, e.Reward, e.NextState, done)
				}
			}

			if done {
				fmt.Printf("Episode %d finished after %d steps. Total Reward: %.2f\n", episode+1, step+1, totalReward)
				break
			}
		}

		totalRewards[episode] = totalReward

		// Early Stopping Check
		if totalReward > bestReward+minImprovement {
			bestReward = totalReward
			waitCounter = 0 // Reset the wait counter since we have improvement
		} else {
			waitCounter++
			if waitCounter >= patience {
				fmt.Println("Early stopping triggered")
				break
			}
		}

		fmt.Printf("Episode %d finished. Total Reward: %.2f\n", episode+1, totalReward)
	}

	fmt.Printf("Episodes finished. Total Episode Reward: %.2f / %.2f\n", sum(totalRewards), maxPossibleTotalRewards)
	// Call the function to visualize the results
	visualizeLearningProgress(totalRewards)
}
