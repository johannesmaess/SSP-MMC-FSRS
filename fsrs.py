import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

plt.style.use('ggplot')

class FsrsSimulator:
    again_cost = 25
    hard_cost = 14
    good_cost = 10
    easy_cost = 6
    first_rating_prob = np.array([0.15, 0.2, 0.6, 0.05])
    review_rating_prob = np.array([0.3, 0.6, 0.1])

    s_min = 0.1
    s_max = 365
    s_eps = 0.4
    s_size = np.ceil((s_max - s_min) / s_eps + 1).astype(int)

    d_min = 1
    d_max = 10
    d_eps = 0.3
    d_size = np.ceil((d_max - d_min) / d_eps + 1).astype(int)

    r_min = 0.69
    r_max = 0.96
    r_eps = 0.03
    r_size = np.ceil((r_max - r_min) / r_eps + 1).astype(int)

    DECAY = -0.5
    FACTOR = 0.9 ** (1.0 / DECAY) - 1.0

    def __init__(self, w, ivl_spacing='equispaced time'):
        ## Init ##
        
        self.w = w
        
        self.cost_matrix = np.zeros((self.d_size, self.s_size))
        self.cost_matrix.fill(1000)
        self.cost_matrix[:, -1] = 0

        self.s_state = np.linspace(self.s_min, self.s_max, self.s_size)
        self.d_state = np.linspace(self.d_min, self.d_max, self.d_size)

        if ivl_spacing == 'equispaced retrievability':
            # option 1: set ivl such that we equally sample retrievability (preserving the original implementation)
            self.r_state = np.linspace(self.r_min, self.r_max, self.r_size)[::-1]
            self.s_state_mesh, self.d_state_mesh, self.r_state_mesh_org = np.meshgrid(self.s_state, self.d_state, self.r_state)
            self.ivl_mesh = self.next_interval(self.s_state_mesh, self.r_state_mesh_org)
        elif ivl_spacing == 'equispaced time':
            # option 2: set ivl such that we equally sample time
            self.ivl_state = np.linspace(1, 365, 365)#*2-1)
            self.s_state_mesh, self.d_state_mesh, self.ivl_mesh = np.meshgrid(self.s_state, self.d_state, self.ivl_state)
        else:
            raise ValueError(f"Invalid ivl_spacing: {ivl_spacing}")

        self.r_state_mesh = self.power_forgetting_curve(self.ivl_mesh, self.s_state_mesh)
        print(self.s_state_mesh.shape, self.d_state_mesh.shape, self.r_state_mesh.shape, self.ivl_mesh.shape)
        
        
    def power_forgetting_curve(self, t, s): # takes in the time since last review and the stability, returns the retrievability
        return (1 + self.FACTOR * t / s) ** self.DECAY


    def next_interval(self, s, r): # takes in stability and (desired) retrievability, returns the interval at which this is achieved
        ivl = s / self.FACTOR * (r ** (1.0 / self.DECAY) - 1.0)
        return np.maximum(1, np.floor(ivl))

    def stability_after_success(self, s, d, r, g):
        return s * (
            1
            + np.exp(self.w[8])
            * (11 - d)
            * np.power(s, -self.w[9])
            * (np.exp((1 - r) * self.w[10]) - 1)
            * (self.w[15] if g == 2 else 1)
            * (self.w[16] if g == 4 else 1)
        )

    def stability_after_failure(self, s, d, r):
        return np.minimum(
            self.w[11]
            * np.power(d, -self.w[12])
            * (np.power(s + 1, self.w[13]) - 1)
            * np.exp((1 - r) * self.w[14]),
            s,
        )

    def mean_reversion(self, init, current):
        return (self.w[7] * init + (1 - self.w[7]) * current).clip(1, 10)


    def next_difficulty(self, d, g):
        return self.mean_reversion(self.w[4], d - self.w[6] * (g - 3))

    # stability to index
    def s2i(self, s): return np.clip(np.floor((s - self.s_min) /
                                        (self.s_max - self.s_min) * self.s_size).astype(int), 0, self.s_size - 1)

    # difficulty to index
    def d2i(self, d): return np.clip(np.floor((d - self.d_min) /
                                        (self.d_max - self.d_min) * self.d_size).astype(int), 0, self.d_size - 1)


    # # retention to index
    # def r2i(r): return np.clip(np.floor((r - r_min) /
    #                                     (r_max - r_min) * r_size).astype(int), 0, r_size - 1)

    # indexes to cost
    def i2c(self, s, d):
        # as s & d are 3D, d2i & s2i are 3D. their first two dimensions match. (they also match cost_matrix but that doen't matter!)
        # thus the indexing is broadcasted: 
        #  - we vectorize over the first two dimensions,
        #  - the 3rd dimension of d2i is used as a row index for cost_matrix, and
        #  - the 3rd dimension of s2i is used as a column index for cost_matrix.
        return self.cost_matrix[self.d2i(d), self.s2i(s)]
    
    def get_init_cost(self):
        init_stability = np.array(self.w[0:4])
        init_difficulty = np.array([self.w[4] - (3 - g) * self.w[5] for g in range(1, 5)])
        init_cost = self.cost_matrix[self.d2i(init_difficulty), self.s2i(init_stability)]
        return init_cost
    
    def get_avg_cost(self):
        return self.get_init_cost() @ self.first_rating_prob
    
    def get_retention_matrix(self):
        ii, jj = np.ogrid[:self.d_size, :self.s_size]
        retention_matrix = self.r_state_mesh[ii,jj,np.argmin(self.cost_matrix_per_ivl, axis=2)]
        return retention_matrix
    
    def run(self, max_iter=1000, minimum_diff_per_iteration=1e-4):
        ## value iteration ##
        
        i = 0
        diff = 1e10

        start = time.time()
        while i < max_iter and diff > minimum_diff_per_iteration * self.s_size * self.d_size:
            next_stability_after_again = self.stability_after_failure(
                self.s_state_mesh, self.d_state_mesh, self.r_state_mesh
            )
            next_difficulty_after_again = self.next_difficulty(self.d_state_mesh, 1)
            next_cost_after_again = (
                self.i2c(next_stability_after_again, next_difficulty_after_again) + self.again_cost
            )

            next_stability_after_hard = self.stability_after_success(
                self.s_state_mesh, self.d_state_mesh, self.r_state_mesh, 2
            )
            next_difficulty_after_hard = self.next_difficulty(self.d_state_mesh, 2)
            next_cost_after_hard = (
                self.i2c(next_stability_after_hard, next_difficulty_after_hard) + self.hard_cost
            )

            next_stability_after_good = self.stability_after_success(
                self.s_state_mesh, self.d_state_mesh, self.r_state_mesh, 3
            )
            next_difficulty_after_good = self.next_difficulty(self.d_state_mesh, 3)
            next_cost_after_good = (
                self.i2c(next_stability_after_good, next_difficulty_after_good) + self.good_cost
            )

            next_stability_after_easy = self.stability_after_success(
                self.s_state_mesh, self.d_state_mesh, self.r_state_mesh, 4
            )
            next_difficulty_after_easy = self.next_difficulty(self.d_state_mesh, 4)
            next_cost_after_easy = (
                self.i2c(next_stability_after_easy, next_difficulty_after_easy) + self.easy_cost
            )

            self.cost_matrix_per_ivl = (
                self.r_state_mesh
                * (
                    self.review_rating_prob[0] * next_cost_after_hard
                    + self.review_rating_prob[1] * next_cost_after_good
                    + self.review_rating_prob[2] * next_cost_after_easy
                )
                + (1 - self.r_state_mesh) * next_cost_after_again
            )
            # update cost matrix
            optimal_cost = self.optimal_cost_matrix()
            diff = self.cost_matrix.sum() - optimal_cost.sum()
            self.cost_matrix = optimal_cost

            if i % 10 == 0:
                print(f"iteration {i:>5}, diff {diff:.2f}, time {time.time() - start:.2f}s")
            i += 1

        end = time.time()
        print(f"Time: {end - start:.2f}s")
        
    def optimal_cost_matrix(self):
        return np.minimum(self.cost_matrix, self.cost_matrix_per_ivl.min(axis=2))