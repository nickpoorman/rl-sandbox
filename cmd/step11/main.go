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
	NumStates  = 5
	NumActions = 2 // 0: move left, 1: move right
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
	Policy  [NumStates]int
	QValues [NumStates][NumActions]float64
	Visits  [NumStates][NumActions]int
	Epsilon float64
}

// func (agent *Agent) Act(state int) int {
// 	if rand.Float64() < agent.Epsilon {
// 		// Exploration: choose a random action
// 		return rand.Intn(NumActions)
// 	}
// 	// Exploitation: choose the best known action
// 	if agent.QValues[state][0] > agent.QValues[state][1] {
// 		return 0
// 	}
// 	return 1
// }

// Here, we’ve replaced the epsilon-greedy strategy with softmax exploration,
// which selects actions based on their estimated values, creating a probability
// distribution where better actions are more likely to be selected. This
// approach encourages exploration of not just random actions but also actions
// that the agent believes are promising.
func (agent *Agent) Act(state int) int {
	var sumExpQ float64
	expQ := make([]float64, NumActions)
	for a := 0; a < NumActions; a++ {
		expQ[a] = math.Exp(agent.QValues[state][a])
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

// UpdateQValues function to update the Q-values based on the observed reward
func (agent *Agent) UpdateQValues(state, action int, reward float64, newState int) {
	learningRate := 1.0 / (1.0 + float64(agent.Visits[state][action]))
	agent.Visits[state][action]++
	discountFactor := 0.99

	bestNextQ := max(agent.QValues[newState][0], agent.QValues[newState][1])
	agent.QValues[state][action] = (1-learningRate)*agent.QValues[state][action] + learningRate*(reward+discountFactor*bestNextQ)
}

func main() {
	agent := Agent{Epsilon: 0.1, Visits: [NumStates][NumActions]int{}}

	for episode := 0; episode < 1000; episode++ {
		env := Environment{}                               // reset environment at the start of each episode
		agent.Epsilon = 1.0 / (1.0 + float64(episode)/100) // decay epsilon

		totalReward := 0.0
		for step := 0; step < 100; step++ {
			state := env.State
			action := agent.Act(state)
			newState, reward, done := env.Step(action)
			agent.UpdateQValues(state, action, reward, newState)
			totalReward += reward

			fmt.Printf("Episode %d, Step %d: State: %d, Action: %d, New State: %d, Reward: %.2f\n", episode+1, step+1, state, action, newState, reward)

			if done {
				fmt.Printf("Episode %d finished after %d steps. Total Reward: %.2f\n", episode+1, step+1, totalReward)
				break
			}
		}

		fmt.Printf("Episode %d finished. Total Reward: %.2f\n", episode+1, totalReward)
	}
}
