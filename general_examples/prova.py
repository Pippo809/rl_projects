import numpy as np

def _discounted_return(rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        list_of_discounted_returns = [sum([0.9**t_prime * reward for t_prime, reward in enumerate(rewards[:])]) for t in range(len(rewards))]
        return list_of_discounted_returns

def _discounted_cumsum(rewards):
    """
        Helper function which
        -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
        -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
    """
    # for t in range(len(rewards)):
    #     print(t)
    #     print(rewards[t:])
    #     for t_prime, reward in enumerate(rewards[t:]):
    #         print(t_prime, reward)
    #         print((t_prime-t))
    #     print([0.9**(t_prime-t) * reward for t_prime, reward in enumerate(rewards[t:])])
    #     print(sum([0.9**(t_prime-t) * reward for t_prime, reward in enumerate(rewards[t:])]))
    #     print("")

    list_of_discounted_cumsums = [sum([0.9**(t_prime) * reward for t_prime, reward in enumerate(rewards[t:])]) for t in range(len(rewards))]

    return list_of_discounted_cumsums

def _discounted_cumsum_2(rewards):
        """
            Input:
                a list of length T
                a list of rewards {r_0, r_1, ..., r_t', ... r_{T-1}} from a single rollout of length T
            Output:
                a list of length T
                a list where the entry in each index t is sum_{t'=t}^{T-1} gamma^(t'-t) * r_{t'}
        """

        all_discounted_cumsums = []

        # for loop over steps (t) of the given rollout
        for start_time_index in range(len(rewards)):

            # 1) create a list of indices (t'): goes from t to T-1
            indices = np.arange(start_time_index, len(rewards))

            # 2) create a list where the entry at each index (t') is gamma^(t'-t)
            discounts = np.power(0.9, indices - start_time_index)

            # 3) create a list where the entry at each index (t') is gamma^(t'-t) * r_{t'}
            # Hint: remember that t' goes from t to T-1, so you should use the rewards from those indices as well
            discounted_rtg = rewards[indices] * discounts

            # 4) calculate a scalar: sum_{t'=t}^{T-1} gamma^(t'-t) * r_{t'}
            sum_discounted_rtg = np.sum(discounted_rtg)

            # appending each of these calculated sums into the list to return
            all_discounted_cumsums.append(sum_discounted_rtg)
        list_of_discounted_cumsums = np.array(all_discounted_cumsums)
        return list_of_discounted_cumsums

if __name__ == "__main__":
    rewards = np.array([1, 1, 1, 1, 1, 1, 1, 1,1,1,1])
    print(_discounted_cumsum(rewards))
    print(_discounted_cumsum_2(rewards))