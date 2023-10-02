import numpy as np

def compute_gae(rewards, values, gamma=0.99, lambda_=0):
    """
    Compute the Generalized Advantage Estimation (GAE) for a given trajectory.

    Args:
        rewards (list or np.ndarray): List of rewards received during the trajectory.
        values (list or np.ndarray): List of state-value estimates for the trajectory.
        gamma (float): Discount factor.
        lambda_ (float): GAE parameter.

    Returns:
        np.ndarray: Array containing the GAE values for each time step in the trajectory.
    """
    T = len(rewards)
    deltas = np.zeros(T)
    advantages = np.zeros(T)
    values.append(0)
    last_advantage = 0

    # Compute advantages in reverse order (from the last time step to the first)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        last_advantage = delta + gamma * lambda_ * last_advantage
        advantages[t] = last_advantage
        deltas[t] = delta

    return advantages

# Example usage:
if __name__ == "__main__":
    # Example rewards and state-value estimates
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5]
    values = [0.5, 0.6, 0.7, 0.8, 0.9]

    # Compute GAE
    advantages = compute_gae(rewards, values)

    # Print the GAE values
    for t, advantage in enumerate(advantages):
        print(f"Time Step {t}: GAE = {advantage}")
