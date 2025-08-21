import math
import numpy as np
from scipy.stats import qmc
from scipy.interpolate import CubicSpline
from scipy.interpolate import BSpline
from scipy.interpolate import make_interp_spline

class ManyTrajGenerator():

    def __init__(self, n_traj, n_params, l_bound, u_bound, seed, knots, coeffs):
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

        self.knots = knots
        self.orig_coeffs = coeffs
    
    
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

        traj = self.gen_traj_from_params(params, self.orig_coeffs, self.knots)

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
    def fit_clamped_bspline_zero_end_va(
        t_real, y, *, degree=3, n_internal_knots=5,
        perturb=False,            # <-- toggle noise on/off
        noise_std=0.0,            # std dev of Gaussian noise (same units as y)
        random_state=None
    ):
        """
        Cubic, clamped B-spline fit with zero velocity/acceleration at both ends.
        - If perturb=True, adds N(0, noise_std^2) to the *middle* (free) control points
        before building the spline (endpoints' zero-derivative constraints preserved).
        - If perturb=False, identical behavior to the original function.
        Returns pos/vel/acc callables in REAL time and the effective control points used.
        """
        p = degree
        if p != 3:
            raise ValueError("This helper assumes cubic splines (degree=3).")
        if n_internal_knots < 3:
            raise ValueError("n_internal_knots must be >= 3 so there are free control points.")

        t_real = np.asarray(t_real, dtype=float)
        y = np.asarray(y, dtype=float)

        # Drop NaNs/Infs and sort by time
        m = np.isfinite(t_real) & np.isfinite(y)
        t_real = t_real[m]; y = y[m]
        order = np.argsort(t_real)
        t_real = t_real[order]; y = y[order]

        if len(t_real) < p + 3:
            raise ValueError("Not enough samples to fit a cubic spline.")

        t0 = t_real[0]
        t1 = t_real[-1]
        T = float(t1 - t0)
        if T <= 0:
            raise ValueError("Time vector must span a positive duration.")
        alpha = 1.0 / T

        # Normalize to u in [0,1]
        u_data = (t_real - t0) * alpha

        # Open/clamped knot vector on [0,1]
        knots_internal = np.linspace(0, 1, n_internal_knots + 2)[1:-1] if n_internal_knots > 0 else np.array([], float)
        t = np.r_[np.zeros(p+1), knots_internal, np.ones(p+1)]
        n_ctrl = len(t) - p - 1  # = n_internal_knots + 4 (for cubic)

        # Basis matrix at data positions
        def basis_col(j):
            coeff = np.zeros(n_ctrl); coeff[j] = 1.0
            return BSpline(t, coeff, p)(u_data)
        A = np.column_stack([basis_col(j) for j in range(n_ctrl)])

        # Tie first three and last three control points to start/end value
        start, end = y[0], y[-1]
        fixed_idx = [0, 1, 2, n_ctrl-3, n_ctrl-2, n_ctrl-1]
        free_idx  = [j for j in range(n_ctrl) if j not in fixed_idx]

        P_fixed = np.array([start, start, start, end, end, end], dtype=float)
        A_fixed = A[:, fixed_idx]
        A_free  = A[:, free_idx]
        rhs = y - A_fixed @ P_fixed

        # Least squares for free control points
        P_free, *_ = np.linalg.lstsq(A_free, rhs, rcond=None)

        # Base (clean) control points
        P = np.empty(n_ctrl, dtype=float)
        P[fixed_idx] = P_fixed
        P[free_idx]  = P_free

        # Optionally perturb only the middle (free) control points
        if perturb and noise_std > 0.0:
            rng = np.random.default_rng(random_state)
            P_eff = P.copy()
            P_eff[free_idx] += rng.normal(0.0, float(noise_std), size=len(free_idx))
        else:
            P_eff = P  # identical to original behavior

        # Build spline (using effective control points) and its derivatives in u-domain
        spline_u = BSpline(t, P_eff, p)
        s1_u = spline_u.derivative(1)
        s2_u = spline_u.derivative(2)

        # Real-time callables with chain rule
        def pos(t_query):
            tq = np.asarray(t_query, dtype=float)
            u = (tq - t0) * alpha
            return spline_u(u)

        def vel(t_query):
            tq = np.asarray(t_query, dtype=float)
            u = (tq - t0) * alpha
            return alpha * s1_u(u)

        def acc(t_query):
            tq = np.asarray(t_query, dtype=float)
            u = (tq - t0) * alpha
            return (alpha**2) * s2_u(u)

        return dict(
            pos=pos, vel=vel, acc=acc,
            t0=t0, T=T, alpha=alpha,
            spline_u=spline_u, s1_u=s1_u, s2_u=s2_u,
            knots=t, ctrl=P_eff  # control points actually used (perturbed or not)
        )


    @staticmethod
    def gen_traj_from_params(params : np.ndarray, orig_coeffs, knots) -> np.ndarray:

        params = params.reshape((-1,6))

        coeffs = orig_coeffs.copy()
        coeffs[3:-3] = params

        spl = BSpline(knots*3, coeffs, 3)

        fine_n_points = 20
        fine_tt = np.linspace(0, 3, fine_n_points)
        fine_qq = spl(fine_tt)
        fine_dqq = spl(fine_tt, 1)

        # import matplotlib.pyplot as plt
        # plt.plot(fine_tt, fine_qq)
        # plt.show()

        # plt.plot(fine_tt, fine_dqq)
        # plt.show()

        # plt.plot(fine_tt, spl(fine_tt, 2))
        # plt.show()

        return fine_tt, fine_qq, fine_dqq



    # @staticmethod
    # def gen_traj_from_params(params : np.ndarray) -> np.ndarray:

    #     T  = params[0]

    #     n_joints = 6
    #     n_points = len(params[1:]) // n_joints + 2

    #     q0 = np.array([math.pi/2, -3*math.pi/4, 0, -math.pi/2, 0, 0 ])
    #     qf = np.array([ 0, -math.pi/4, math.pi/4, 0, math.pi/4, math.pi/2 ])

    #     tt = np.linspace(0, T, n_points)

    #     qq = np.reshape(params[1:], (n_points - 2, n_joints))
    #     qq = np.vstack((q0, qq, qf))

    #     # Cubic spline with zero velocity at start & end
    #     cs = CubicSpline(tt, qq, bc_type='clamped', axis=0)

    #     fine_n_points = 20
    #     fine_tt = np.linspace(0, T, fine_n_points)
    #     fine_qq = cs(fine_tt)
    #     fine_dqq = cs(fine_tt, 1)
        
    #     # import matplotlib.pyplot as plt
    #     # plt.plot(fine_tt, fine_qq)
    #     # plt.show()

    #     # plt.plot(fine_tt, fine_dqq)
    #     # plt.show()

    #     return fine_tt, fine_qq, fine_dqq


    

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

        q0 = np.array([math.pi/2, -3*math.pi/4, 0, -math.pi/2, 0, 0 ])
        qf = np.array([ 0, -math.pi/4, math.pi/4, 0, math.pi/4, math.pi/2 ])
        T = 3

        times = np.array([0, T])
        positions = np.vstack((q0, qf))

        spl = make_interp_spline(times, positions, axis=0, k=5,
                         bc_type=([(1,np.zeros_like(q0)),(2,np.zeros_like(q0))],
                                  [(1,np.zeros_like(qf)),(2,np.zeros_like(qf))]))
        
        ftt = np.linspace(0, T, 100)
        
        bspline_models = []
        coeffs = []
        

        for i in range(6):
            model = ManyTrajGenerator.fit_clamped_bspline_zero_end_va(ftt, spl(ftt)[:,i],
                                            perturb=False, n_internal_knots=7)
            bspline_models.append(model)
            coeffs.append(model['ctrl'])
        
        coeffs = np.array(coeffs).T

        knots = bspline_models[0]["knots"]

        traj_params = coeffs[3:-3]

        traj_var = 0.05

        # print(traj_params)

        traj_param_min = traj_params - traj_var
        traj_param_max = traj_params + traj_var

        # n_points = 7

        # tt = np.linspace(0, T, n_points)

        # # Cubic spline with zero velocity at start & end
        # cs = CubicSpline(times, positions, bc_type='clamped', axis=0)

        # qq = cs(tt)
        # qq = qq[1:-1]

        # traj_var = 0.1

        # traj_param_min = qq - traj_var
        # traj_param_max = qq + traj_var

        lb = traj_param_min.ravel()
        ub = traj_param_max.ravel()

        return lb, ub, knots, coeffs




if __name__ == "__main__":
    lb, ub, knots, coeffs = ManyTrajGenerator.generate_param_bounds()

    # print(lb.shape)
    # print(ub.shape)
    # print(knots.shape)
    # print(coeffs.shape)

    mt_gen = ManyTrajGenerator(n_traj = 5, n_params = 30, l_bound=lb, u_bound=ub, seed=42, knots = knots, coeffs = coeffs)

    print(mt_gen.get_i_traj(0)[1])
    # print(mt_gen.get_i_traj(3)[1])

    # mt_gen.generate_param_bounds()
