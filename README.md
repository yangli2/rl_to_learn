# rl_to_learn

Learning is a reinforcement-learning problem.

1. We measure our states: current weights, current gradients, maybe histories, from sampled minibatches.
1. We evaluate our policy based on the states: the policy network. We get \Delta w for the value network.
1. We sample the reward (i.e. loss) while using w + \Delta w as the weights of the value network.
1. We do gradient ascent on the policy network weights.
1. We apply the \Delta w to the value network weights.

