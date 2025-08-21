import math
import numpy as np
from scipy.stats import qmc
from scipy.interpolate import CubicSpline

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
    
    def get_all_params(self):
        return {"samples": self.sample.flatten().tolist(),
                "l_bound": self.l_bound.flatten().tolist(),
                "u_bound": self.u_bound.flatten().tolist(),
                "n_traj": self.n_traj, "seed": self.seed,
                "samples_shape": list(self.sample.shape)}
    

    def get_i_traj(self, idx):
        norm_params = self.sample[idx]
        params = self.l_bound + (self.u_bound - self.l_bound) * norm_params

        traj = self.gen_traj_from_params(params)

        return traj
    

    # @staticmethod
    # def gen_traj_from_params(params: np.ndarray) -> np.ndarray:
    #     # assume that params modifies the trajectory

    #     t_traj = np.array([0.0, 3.0])
    #     pos = np.array([[math.pi/2, -3*math.pi/4, 0, -math.pi/2, 0, 0 ],
    #                     [ 0, -math.pi/4, math.pi/4, 0, math.pi/4, math.pi/2 ]])
    #     vel = np.zeros((2,6), dtype=float)

    #     return t_traj, pos, vel


    @staticmethod
    def gen_traj_from_params(params : np.ndarray) -> np.ndarray:

        T  = params[0]

        n_joints = 6
        n_points = len(params[1:]) // n_joints + 2

        q0 = np.array([math.pi/2, -3*math.pi/4, 0, -math.pi/2, 0, 0 ])
        qf = np.array([ 0, -math.pi/4, math.pi/4, 0, math.pi/4, math.pi/2 ])

        tt = np.linspace(0, T, n_points)

        qq = np.reshape(params[1:], (n_points - 2, n_joints))
        qq = np.vstack((q0, qq, qf))

        # Cubic spline with zero velocity at start & end
        cs = CubicSpline(tt, qq, bc_type='clamped', axis=0)

        fine_n_points = 20
        fine_tt = np.linspace(0, T, fine_n_points)
        fine_qq = cs(fine_tt)
        fine_dqq = cs(fine_tt, 1)
        
        # import matplotlib.pyplot as plt
        # plt.plot(fine_tt, fine_qq)
        # plt.show()

        # plt.plot(fine_tt, fine_dqq)
        # plt.show()

        return fine_tt, fine_qq, fine_dqq


    

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


    @staticmethod
    def generate_param_bounds():
        T_param_mean = 3
        T_param_var = 0.5
        T_param_min = T_param_mean - T_param_var
        T_param_max = T_param_mean + T_param_var

        q0 = np.array([math.pi/2, -3*math.pi/4, 0, -math.pi/2, 0, 0 ])
        qf = np.array([ 0, -math.pi/4, math.pi/4, 0, math.pi/4, math.pi/2 ])
        T = 3

        n_points = 7

        tt = np.linspace(0, T, n_points)

        times = np.array([0, T])
        positions = np.vstack((q0, qf))

        # Cubic spline with zero velocity at start & end
        cs = CubicSpline(times, positions, bc_type='clamped', axis=0)

        qq = cs(tt)
        qq = qq[1:-1]

        traj_var = 0.1

        traj_param_min = qq - traj_var
        traj_param_max = qq + traj_var

        lb = np.concatenate([np.array([T_param_min]), traj_param_min.ravel()])
        ub = np.concatenate([np.array([T_param_max]), traj_param_max.ravel()])

        return lb, ub




if __name__ == "__main__":
    lb, ub = ManyTrajGenerator.generate_param_bounds()

    mt_gen = ManyTrajGenerator(n_traj = 5, n_params = 31, l_bound=lb, u_bound=ub, seed=42)

    print(mt_gen.get_i_traj(0)[1])
    # print(mt_gen.get_i_traj(3)[1])

    # mt_gen.generate_param_bounds()
