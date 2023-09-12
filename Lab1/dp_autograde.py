import numpy as np
from collections import defaultdict

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    while True:
        delta = 0  # Initialize the change in value function

        for s in range(env.nS):
            v_s = 0  # Initialize the new value of state s

            # Calculate the expected value of state s under the current policy
            for a, phi_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v_s += phi_prob * prob * (reward + discount_factor * V[next_state])

            # Calculate the change in the value function for this state
            delta = max(delta, abs(V[s] - v_s))

            # Update the value function for state s
            V[s] = v_s

        # If the change in value function is smaller than theta, stop
        if delta < theta:
            break

    return np.array(V)

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    # # YOUR CODE HERE
    while True:
        # Evaluate the current policy
        V = policy_eval_v(policy, env, discount_factor)
        
        # Policy improvement
        policy_stable = True
        
        for s in range(env.nS):
            old_action = np.argmax(policy[s])
            
            # Compute action values to determine the best action
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, _ in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            
            # Select the best action
            best_action = np.argmax(action_values)
            policy[s] = np.eye(env.nA)[best_a]
            
            # Check if policy has changed
            if old_action != best_a:
                policy_stable = False
                
        # If the policy is stable (no changes), we're done
        if policy_stable:
            break
        
    # END YOUR CODE
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    # # YOUR CODE HERE
    while True:
        # Evaluate the current policy
        V = policy_eval_v(policy, env, discount_factor)
        
        # Policy improvement
        policy_stable = True
        
        for s in range(env.nS):
            old_action = np.argmax(policy[s])
            
            # Compute action values to determine the best action
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, _ in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            
            # Select the best action
            best_action = np.argmax(action_values)
            policy[s] = np.eye(env.nA)[best_a]
            
            # Check if policy has changed
            if old_action != best_a:
                policy_stable = False
                
        # If the policy is stable (no changes), we're done
        if policy_stable:
            break
        
    # END YOUR CODE
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    # # YOUR CODE HERE
    while True:
        # Evaluate the current policy
        V = policy_eval_v(policy, env, discount_factor)
        
        # Policy improvement
        policy_stable = True
        
        for s in range(env.nS):
            old_action = np.argmax(policy[s])
            
            # Compute action values to determine the best action
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, _ in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            
            # Select the best action
            best_action = np.argmax(action_values)
            policy[s] = np.eye(env.nA)[best_action]
            
            # Check if policy has changed
            if old_action != best_a:
                policy_stable = False
                
        # If the policy is stable (no changes), we're done
        if policy_stable:
            break
        
    # END YOUR CODE
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    # # YOUR CODE HERE
    while True:
        # Evaluate the current policy
        V = policy_eval_v(policy, env, discount_factor)
        
        # Policy improvement
        policy_stable = True
        
        for s in range(env.nS):
            old_action = np.argmax(policy[s])
            
            # Compute action values to determine the best action
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, _ in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            
            # Select the best action
            best_action = np.argmax(action_values)
            policy[s] = np.eye(env.nA)[best_action]
            
            # Check if policy has changed
            if old_action != best_action:
                policy_stable = False
                
        # If the policy is stable (no changes), we're done
        if policy_stable:
            break
        
    # END YOUR CODE
    return policy, V

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    # YOUR CODE HERE
    raise NotImplementedError
    return policy, Q

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    # YOUR CODE HERE

    # Repeat until convergence
    while True:
        
        # Initialize the change in Q-values to zero
        delta = 0
        
        # Loop over all states and actions
        for s in range(env.nS):
            for a in range(env.nA):
                
                # Calculate the new Q-value for this state-action pair
                new_q_value = 0
                for prob, next_state, reward, done in env.P[s][a]:
                    new_q_value += prob * (reward + discount_factor * np.max(Q[next_state]))
                
                # Update the change in Q-values
                delta = max(delta, abs(new_q_value - Q[s][a]))
                
                # Update the Q-value for this state-action pair
                Q[s][a] = new_q_value
        
        # Check if the change in Q-values is below the threshold
        if delta < theta:
            break

    # Extract the optimal policy
    policy = np.zeros((env.nS, env.nA))
    
    # Loop over all states and actions and find the optimal action for each state
    for s in range(env.nS):
        best_action = np.argmax(Q[s])
        policy[s][best_action] = 1.0

    return policy, Q
