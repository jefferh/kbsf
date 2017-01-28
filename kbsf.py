from __future__ import division
from math import *
import numpy as np
import random

class KBSF_Inputs:
    """
    
    Attributes:
    sample_transitions = dictionary containing the observed transitions 
                         and rewards; keys are actions, values are lists 
                         of 3-tuples corresponding to the samples (initial 
                         state, reward, terminal state) for that action
    rep_states = list of representative states
    kbrl_mo_kernel = "mother kernel" used to construct the "K" matrices
    kbrl_bandwidth = bandwidth used to construct the "K" matrices
    kbrl_dist = distance function to be used with kbrl_mo_kernel
    kbsf_mo_kernel = "mother kernel" used to construct the "D" matrices
    kbsf_bandwidth = bandwidth used to construct the "D" matrices
    kbsf_norm = distance function to be used with kbsf_mo_kernel
    discount = discount factor 
    N = total number of samples
    M = total number of representative states
    """
    def __init__(self, S, S_bar, phi, tau, dist,
                     phi_bar, tau_bar, dist_bar, gamma):

        self.sample_transitions = S
        self.rep_states = S_bar
        self.kbrl_mo_kernel = phi
        self.kbrl_bandwidth = tau
        self.kbrl_dist = dist
        self.kbsf_mo_kernel = phi_bar
        self.kbsf_bandwidth = tau_bar
        self.kbsf_dist = dist_bar
        self.discount = gamma
        """Compute the total number of samples."""
        count = 0
        for action in self.sample_transitions:
            count += len(self.sample_transitions[action])
        self.N = count
        self.M = len(self.rep_states)
                
    def samples(self, action):
        """Return the list of samples for a given action."""
        return self.sample_transitions[action]

    def n(self, action):
        """Return the number of samples associated with a given action."""
        return len(self.sample_transitions[action])

    def s(self, a, i):
        """Return the ith initial state in the samples for action a."""
        return self.sample_transitions[a][i][0]

    def r(self, a, i):
        """Return the ith observed reward in the samples for action a."""
        return self.sample_transitions[a][i][1]

    def s_hat(self, a, i):
        """Return the ith terminal state in the samples for action a."""
        return self.sample_transitions[a][i][2]

    def kbrl_kernel(self, s1, s2):
        """Return the value of the KBRL kernel for states s1 and s2."""
        return self.kbrl_mo_kernel(self.kbrl_dist(s1, s2) / self.kbrl_bandwidth)

    def kbrl_norm_kernel(self, s, a, j):
        """Return the value of the normalized KBRL kernel for state s and the jth 
           initial state for action a."""
        normalizer = 0
        for i in range(self.n(a)):
            normalizer += self.kbrl_kernel(s, self.s(a, i))
        return self.kbrl_kernel(s, self.s(a, j)) / normalizer

    def kbsf_kernel(self, s1, s2):
        """Return the value of the KBSF kernel for states s1 and s2."""
        return self.kbsf_mo_kernel(self.kbsf_dist(s1, s2) / self.kbsf_bandwidth)

    def kbsf_norm_kernel(self, s, j):
        """Return the value of the normalized KBSF kernel for state s and the jth 
           representative state."""
        normalizer = 0
        for i in range(self.M):
            normalizer += self.kbsf_kernel(s, self.rep_states[i])
        return self.kbsf_kernel(s, self.rep_states[j]) / normalizer

class MDP:
    def __init__(self, mdp_dict):
        """
        Input: a dictionary representation of the MDP; the keys are state-action 
               pairs, and the values are pairs whose first element is the 
               corresponding one-step reward and whose second element is a 
               dictionary where each key is a state and each value is the 
               probability of transitioning to that state

        Attributes:
        m = number of state-action pairs
        n = number of states
        X = list of states
        A = dictionary of lists of available actions; keys are states, values are 
            sets of actions
        r = dictionary of one-step rewards; keys are state-action pairs, values 
            are rewards
        p = dictionary of transition probabilities; keys are state-action pairs, 
            values are dictionaries where each key is a state and the corresponding 
            value is the probability of transitioning to that state
        """
        self.m = len(mdp_dict)
        
        states = set()
        for state_action_pair in mdp_dict.keys():
            states.add(state_action_pair[0])
        self.n = len(states)        
        self.X = list(states)

        actions = {}
        for state in states:
            act_set = set()
            for key in mdp_dict:
                if key[0] == state:
                    act_set.add(key[1])
            actions[state] = list(act_set)
        self.A = actions

        rewards = {}
        for state_action_pair in mdp_dict.keys():
            rewards[state_action_pair] = mdp_dict[state_action_pair][0]
        self.r = rewards

        transition_probs = {}
        for state_action_pair in mdp_dict.keys():
            transition_probs[state_action_pair] = mdp_dict[state_action_pair][1]
        self.p = transition_probs

def policy_eval(pi, mdp, discount_factor):
    P_pi = np.fromfunction(np.vectorize(lambda i,j : mdp.p[mdp.X[i],
               pi[mdp.X[i]]][mdp.X[j]]), (mdp.n, mdp.n), dtype=int)
    r_pi = np.fromfunction(np.vectorize(lambda i,j : mdp.r[mdp.X[i],
               pi[mdp.X[i]]]), (mdp.n, 1), dtype=int)
    return np.linalg.inv(np.eye(mdp.n) - discount_factor * P_pi).dot(r_pi)

def greedy_action(state, v, mdp, discount_factor):    
    state_action_values = [(mdp.r[state, action] +
                            sum([mdp.p[state, action][mdp.X[j]] * v[j]
                                for j in range(len(mdp.X))]))[0]
                            for action in mdp.A[state]]
    return mdp.A[state][np.argmax(state_action_values)]

def policy_iteration(mdp, discount_factor):
    """Select an initial policy."""
    pi = {state : random.choice(mdp.A[state]) for state in mdp.X}
    while True:
        """Evaluate the current policy."""
        v_pi = policy_eval(pi, mdp, discount_factor)
        """Try to improve the current policy."""
        updated = False
        for state in mdp.X:
            a_star = greedy_action(state, v_pi, mdp, discount_factor)
            if a_star != pi[state]:
                pi[state] = a_star
                updated = True
        if not(updated):
            return pi
        
def construct_mdp(inputs):
    D_matrices = {}
    K_matrices = {}
    r_vectors = {}
    P_matrices = {}
    for action in inputs.sample_transitions:
        """Compute the matrix \dot{D}^a."""
        D_matrices[action] = np.fromfunction(np.vectorize(lambda i,j :
            inputs.kbsf_norm_kernel(inputs.s_hat(action, i), j)),
            (inputs.n(action), inputs.M), dtype=int)
        """Compute the matrix \dot{K}^a."""
        K_matrices[action] = np.fromfunction(np.vectorize(lambda i,j :
            inputs.kbrl_norm_kernel(inputs.rep_states[i], action, j)),
            (inputs.M, inputs.n(action)), dtype=int)
        """Compute the one-step reward vector \bar{r}^a."""
        r_a = np.array([inputs.r(action, i) for i in range(inputs.n(action))])
        r_vectors[action] = K_matrices[action].dot(r_a)
        """Compute the transition matrix \bar{P}^a."""
        P_matrices[action] = K_matrices[action].dot(D_matrices[action])
    """Construct the finite MDP."""
    mdp_dict = {}
    for i in range(inputs.M):
        for action in inputs.sample_transitions:
            """Fetch the transition probabilities."""
            prob_dict = dict()
            for j in range(inputs.M):
                prob_dict[inputs.rep_states[j]] = P_matrices[action][i][j]
            mdp_dict[(inputs.rep_states[i], action)] = (r_vectors[action][i],
                                                            prob_dict)
    """
    The matrices \dot{D}^a are needed to extend the value function of the KBSF MDP
    to an approximate value function for the KBRL MDP.
    """
    return (mdp_dict, D_matrices)

def compute_apx_kbrl_value(inputs):
    """
    Compute the KBSF approximation of the value function of the KBRL MDP.
    """
    """Construct the KBSF MDP."""
    (kbsf_mdp_dict, D_matrices) = construct_mdp(inputs)
    kbsf_mdp = MDP(kbsf_mdp_dict)
    """Compute the value function for the KBSF MDP."""
    kbsf_pi = policy_iteration(kbsf_mdp, inputs.discount)
    kbsf_v = policy_eval(kbsf_pi, kbsf_mdp, inputs.discount)
    """Extend the value function for the KBSF MDP to a function on the KBRL MDP's 
       state set."""
    # D = np.concatenate([D_matrices[action] for action in D_matrices])
    def v_tilde(a, i):
        return sum([inputs.kbsf_norm_kernel(inputs.s_hat(a, i), j) * kbsf_v[j][0]
                        for j in range(kbsf_mdp.n)])
    return v_tilde
    
def kbsf_q_func(inputs, s, a):
    v_tilde = compute_apx_kbrl_value(inputs)
    return sum([inputs.kbrl_norm_kernel(s, a, i) * (inputs.r(a, i) + inputs.discount * v_tilde(a, i))
                    for i in range(inputs.n(a))])

def kbsf_policy(inputs):
    def pi(state):
        d = {action : kbsf_q_func(inputs, state, action) for action in inputs.sample_transitions.keys()}
        return max(d.iterkeys(), key=(lambda key : d[key]))
    return pi
