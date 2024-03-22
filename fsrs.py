import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

plt.style.use('ggplot')

class RangeConfig:
    def __init__(self, min_val, max_val, eps):
        self.min = min_val
        self.max = max_val
        self.eps = eps
        self.size = self._calculate_size()

    def _calculate_size(self):
        return np.ceil((self.max - self.min) / self.eps + 1).astype(int)

class FsrsSimulator:
    def __init__(self, w, config_path=None, **kwargs):
        # Read and apply constants from YAML file
        if config_path:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            self.__dict__.update(config)
            
        # Override any instance variables with kwargs
        self.__dict__.update(kwargs)
        
        # Convert s, d, r, ivl to RangeConfig objects
        for key in ['s', 'd', 'r', 'ivl']:
            if hasattr(self, key): setattr(self, key, RangeConfig(*getattr(self, key)))
                
        # Check for required parameters
        required = ['again_cost', 'hard_cost', 'good_cost', 'easy_cost', 'first_rating_prob', 'review_rating_prob', 'DECAY']
        for key in required:
            if not hasattr(self, key):
                raise ValueError(f"Missing required parameter: {key}")

        # calculate FACTOR from DECAY
        self.FACTOR = 0.9 ** (1.0 / self.DECAY) - 1.0
        
        # Check if either r or ivl is provided
        assert hasattr(self, 'r') != hasattr(self, 'ivl'), "Either 'r' or 'ivl' must be provided in config & kwargs."
        
        ## Init variable ranges & meshs ##
        
        self.s_state = np.linspace(self.s.min, self.s.max, self.s.size)
        self.d_state = np.linspace(self.d.min, self.d.max, self.d.size)

        if hasattr(self, 'r'):
            # option 1: set ivl such that we equispaced sample retrievability (preserving the original implementation)
            self.r_state = np.linspace(self.r.min, self.r.max, self.r.size)[::-1]
            self.s_state_mesh, self.d_state_mesh, self.r_state_mesh_org = np.meshgrid(self.s_state, self.d_state, self.r_state)
            self.ivl_mesh = self.next_interval(self.s_state_mesh, self.r_state_mesh_org)
        elif hasattr(self, 'ivl'):
            # option 2: set ivl such that we equispaced sample time
            self.ivl_state = np.linspace(1, 365, 365)#*2-1)
            self.s_state_mesh, self.d_state_mesh, self.ivl_mesh = np.meshgrid(self.s_state, self.d_state, self.ivl_state)
        
        self.r_state_mesh = self.power_forgetting_curve(self.ivl_mesh, self.s_state_mesh)
        
        ## Init weights ##
        
        self.w = w
        
        ## Init cost matrix ##
        
        self.init_cost_matrix()
        
    def init_cost_matrix(self):
        self.cost_matrix = np.zeros((self.d.size, self.s.size))
        self.cost_matrix.fill(1000)
        self.cost_matrix[:, -1] = 0
        
        self.cost_matrix_per_ivl = None
        self.num_iter = 0
        
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
    def s2i(self, s): return np.clip(np.floor((s - self.s.min) /
                                        (self.s.max - self.s.min) * self.s.size).astype(int), 0, self.s.size - 1)

    # difficulty to index
    def d2i(self, d): return np.clip(np.floor((d - self.d.min) /
                                        (self.d.max - self.d.min) * self.d.size).astype(int), 0, self.d.size - 1)


    # # retention to index
    # def r2i(r): return np.clip(np.floor((r - r.min) /
    #                                     (r.max - r.min) * r.size).astype(int), 0, r.size - 1)

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
        ii, jj = np.ogrid[:self.d.size, :self.s.size]
        retention_matrix = self.r_state_mesh[ii,jj,np.argmin(self.cost_matrix_per_ivl, axis=2)]
        return retention_matrix
    
    def run(self, max_iter=1000, minimum_diff_per_iteration=1e-4):
        ## value iteration ##
        diff = 1e10

        start = time.time()
        while self.num_iter < max_iter and diff > minimum_diff_per_iteration * self.s.size * self.d.size:
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

            if self.num_iter % 10 == 0:
                print(f"iteration {self.num_iter:>5}, diff {diff:.2f}, time {time.time() - start:.2f}s")
            self.num_iter += 1

        end = time.time()
        print(f"Time: {end - start:.2f}s")
        
    def optimal_cost_matrix(self):
        return np.minimum(self.cost_matrix, self.cost_matrix_per_ivl.min(axis=2))
    
    
class FsrsSimulatorStorable(FsrsSimulator):
    def __init__(self, w, config_path, npy_path=None):
        self.representative_config_path = config_path
        super().__init__(w=w, config_path=config_path)
        if npy_path: self.load(npy_path)
    
    def get_filename(self):
        w_str = '_'.join([f'{round(w, 2):1.2f}' for w in self.w])
        config_str = self.representative_config_path.split('/')[-1].replace('.yaml', '')
        
        fn = f"{config_str}__{w_str}__{self.num_iter}"
        return fn
    
    def get_path_npy(self):
        return f"{self.get_filename()}.npy"
        
    def save(self):
        np.save(self.get_path_npy(), self.cost_matrix_per_ivl)
    
    def load(self, path):
        # check that path matches the config (except for the num_iter part of the filename)
        assert path.endswith('.npy'), "Path must be a .npy file."
        fn = path.split('/')[-1].replace('.npy', '')
        num_iter = int(fn.split('__')[-1])
        fn_config = fn.replace(f"__{num_iter}", '')
        assert self.get_filename().startswith(fn_config), "Path does not match the config."
        
        # set the cost matrices and num_iter from path
        self.cost_matrix_per_ivl = np.load(path)
        self.cost_matrix = self.optimal_cost_matrix()
        self.num_iter = num_iter
        
    def run(self, *args, save=True, **kwargs):
        super().run(*args, **kwargs)
        if save: self.save()