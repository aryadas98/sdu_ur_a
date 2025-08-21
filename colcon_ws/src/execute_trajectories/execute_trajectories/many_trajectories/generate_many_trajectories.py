import math
import numpy as np
from scipy.stats import qmc

class ManyTrajGenerator():

    def __init__(self, n_traj, n_params, l_bound, u_bound, seed):
        # validate the input
        assert isinstance(l_bound, np.ndarray)
        assert isinstance(u_bound, np.ndarray)
        assert len(l_bound.shape) == 1 and l_bound.shape[0] == n_params
        assert len(u_bound.shape) == 1 and u_bound.shape[0] == n_params
        assert np.all(u_bound >= l_bound)

        rng = np.random.default_rng(seed)
        
        sampler = qmc.Sobol(d=n_params, seed=rng)
        self.sample = sampler.random(n=n_traj)
        
        self.n_traj = n_traj
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.seed = seed
    
    
    def get_i_params(self, idx):
        norm_params = self.sample[idx]
        params = self.l_bound + (self.u_bound - self.l_bound) * norm_params

        return params
    

    def get_i_traj(self, idx):
        norm_params = self.sample[idx]
        params = self.l_bound + (self.u_bound - self.l_bound) * norm_params

        traj = self.gen_traj_from_params(params)

        return traj
    

    @staticmethod
    def gen_traj_from_params(params: np.ndarray) -> np.ndarray:
        # assume that params modifies the trajectory

        t_traj = np.array([0.0, 3.0])
        pos = np.array([[math.pi/2, -3*math.pi/4, 0, -math.pi/2, 0, 0 ],
                        [ 0, -math.pi/4, math.pi/4, 0, math.pi/4, math.pi/2 ]])
        vel = np.zeros((2,6), dtype=float)

        return t_traj, pos, vel

    

    # @staticmethod
    # # override this function
    # def gen_traj_from_params(params : np.ndarray) -> np.ndarray:
    #     ## generates position, velocity curves from params
    #     ## demo method here

    #     q0 = params[0]  # inital point
    #     qf = params[1]  # final point
    #     tf = params[2]  # final time

    #     v_max = 2*(qf-q0)/tf

    #     N = 3
    #     t_traj = np.array([0, tf/2, tf])
    #     q0_traj = np.array([q0, (q0+qf)/2, qf])
    #     dq0_traj = np.array([0, v_max, 0])

    #     HOME = np.array([0, math.radians(-111.26), math.radians(112.08),
    #             math.radians(269.33), math.radians(-89.87), math.radians(95.92)])

    #     HOME_VEL = np.zeros((6,))

    #     pos = np.hstack((q0_traj.reshape((N,1)), np.repeat(HOME[1:].reshape((1,-1)), N, 0)))
    #     vel = np.hstack((dq0_traj.reshape((N,1)), np.repeat(HOME_VEL[1:].reshape((1,-1)), N, 0)))

    #     return t_traj, pos, vel




if __name__ == "__main__":
    l_bound = np.array([-0.2, math.pi/2-0.2, 3-0.2])
    u_bound = np.array([0.2, math.pi/2+0.2, 3+0.2])

    mt_gen = ManyTrajGenerator(n_traj = 5, n_params = 3, l_bound=l_bound, u_bound=u_bound, seed=42)

    print(mt_gen.get_i_traj(0)[1])
    print(mt_gen.get_i_traj(3)[1])
