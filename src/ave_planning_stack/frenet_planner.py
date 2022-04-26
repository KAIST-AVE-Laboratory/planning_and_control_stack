from typing import Any, Dict, List, Tuple

import numpy as np

from ave_planning_stack.polynomials import QuinticPolynomial, QuarticPolynomial


class FrenetPlanner:
    """Optimal Trajectory Generation for Dynamics Street Scenarios in a Frenet Frame."""

    def __init__(self,
        dt: float,
        n_dt: int,
        dd: float,
        n_dd: int,
        kj: float,
        kt: float,
        kd: float,
        k_lat: float,
        k_lon: float) -> None:
        """Constructor.

        Args:
            dt (float): Deviation from T for search space.
            n_dt (int): Number of dt to be sampled in + and - direction.
            dd (float): Deviation from D for search space.
            n_dd (int): Number of dd to be sampled in + and - direction.
            kj (float): Weight for jerk cost.
            kt (float): Weight for time cost.
            kd (float): Weight for deviation from target cost.
            k_lat (float): Weight for lateral path cost.
            k_lon (float): Weight for longitudinal path cost.
        """
        self.dt, self.n_dt = dt, n_dt
        self.dd, self.n_dd = dd, n_dd

        self.kj, self.kt, self.kd = kj, kt, kd
        self.k_lat, self.k_lon = k_lat, k_lon
    

    def plan_distance_keeping(self,
        c_d: Tuple[float, float, float],
        c_s: Tuple[float, float, float],
        t_s: Tuple[float, float, float],
        ds: float,
        n_ds: int,
        d0: float,
        tw: float) -> Dict[str, List[Any]]:
        """Generate candidate plans in the frenet frame for distance keeping with constant time gap.

        This planning method assumes the leading object has zero jerk and is used for car following
        and stopping behavior. See Sec V.A of the paper.

        Args:
            c_d (Tuple[float, float, float]): Current (d, d_dot, d_ddot).
            c_s (Tuple[float, float, float]): Current (s, s_dot, s_ddot).
            t_s (Tuple[float, float, float]): Leading object (s, s_dot, s_ddot).
            ds (float): Deviation from S for search space.
            n_ds (int): Number of ds to be sampled in + and - direction.
            d0 (float): Distance threshold to the leading object.
            tw (float): Time gap to the leading object.

        Returns:
            Dist[str, List[Any]]: Dictionary with a paired keys=("cost", "path").
        """
        tspace = 1 + np.arange(-self.n_dt * self.dt, (self.n_dt + 1) * self.dt - 1e-6, self.dt)
        dspace = np.arange(-self.n_dd * self.dd, (self.n_dd + 1) * self.dd - 1e-6, self.dd)
        sspace = np.arange(-n_ds * ds, (n_ds + 1) * ds - 1e-6, ds)

        paths = {"cost": [], "path": []}

        ## iterate through ts
        for ti in tspace:
            t_ticks = [t for t in np.arange(0, ti, self.dt)]
            lat_plans = []
            lon_plans = []

            ## generate lateral plan (quintic polynom) given current sampled t and d
            for di in dspace:
                lat = QuinticPolynomial(c_d[0], c_d[1], c_d[2], di, 0., 0., ti)

                ## sample exact points along the quintic polynom
                d_pos, d_vel, d_acc, d_jrk = [], [], [], []
                for t in t_ticks:
                    d_pos.append(lat.calc_point(t))
                    d_vel.append(lat.calc_first_derivative(t))
                    d_acc.append(lat.calc_second_derivative(t))
                    d_jrk.append(lat.calc_third_derivative(t))
                dcost = self.kj * sum(np.power(d_jrk, 2)) + self.kt * ti + self.kd * (d_pos[-1] ** 2)

                lat_plans.append({"pos": d_pos, "vel": d_vel, "acc": d_acc, "cost": dcost})

            ## generate target in s-axis for longitudinal plan
            t_s_pos = t_s[0] - d0 - tw * t_s[1]
            t_s_vel = t_s[1] - tw * t_s[2]
            t_s_acc = t_s[2]

            ## generate longitudinal plan (quintic polynom) given current sampled t and s
            for si in sspace:
                lon = QuinticPolynomial(c_s[0], c_s[1], c_s[2], t_s_pos + si, t_s_vel, t_s_acc, ti)

                ## sample exact points along the quintic polynom
                s_pos, s_vel, s_acc, s_jrk = [], [], [], []
                for t in t_ticks:
                    s_pos.append(lon.calc_point(t))
                    s_vel.append(lon.calc_first_derivative(t))
                    s_acc.append(lon.calc_second_derivative(t))
                    s_jrk.append(lon.calc_third_derivative(t))
                scost = self.kj * sum(np.power(s_jrk, 2)) + self.kt * ti + self.kd * (si ** 2)

                lon_plans.append({"pos": s_pos, "vel": s_vel, "acc": s_acc, "cost": scost})
            
            ## combination of each lateral and longitudinal plans
            for lat_plan in lat_plans:
                for lon_plan in lon_plans:
                    paths["path"].append((
                        lat_plan["pos"], lat_plan["vel"], lat_plan["acc"],
                        lon_plan["pos"], lon_plan["vel"], lon_plan["acc"]
                    ))
                    paths["cost"].append(self.k_lat * lat_plan["cost"] + self.k_lon * lon_plan["cost"])
        
        return paths

    
    def velocity_tracking(self,
        c_d: Tuple[float, float, float],
        c_s: Tuple[float, float, float],
        t_s: Tuple[float, float],
        ds: float,
        n_ds: int) -> Dict[str, List[Any]]:
        """Generate candidate plans in the frenet frame for distance keeping with constant time gap.

        This planning method assumes the leading object has zero jerk and is used for velocity keeping
        behavior. See Sec V.B of the paper.

        Args:
            c_d (Tuple[float, float, float]): Current (d, d_dot, d_ddot).
            c_s (Tuple[float, float, float]): Current (s, s_dot, s_ddot).
            t_s (Tuple[float, float]): Target (s_dot, s_ddot).
            ds (float): Deviation from S_dot for search space.
            n_ds (int): Number of ds to be sampled in + and - direction.

        Returns:
            Dist[str, List[Any]]: Dictionary with a paired keys=("cost", "path").
        """
        tspace = 2 + np.arange(-self.n_dt * self.dt, (self.n_dt + 1) * self.dt - 1e-6, self.dt)
        dspace = np.arange(-self.n_dd * self.dd, (self.n_dd + 1) * self.dd - 1e-6, self.dd)
        sspace = np.arange(-n_ds * ds, (n_ds + 1) * ds - 1e-6, ds)

        paths = {"cost": [], "path": []}

        ## iterate through ts
        for ti in tspace:
            t_ticks = [t for t in np.arange(0, ti, self.dt)]
            lat_plans = []
            lon_plans = []

            ## generate lateral plan (quintic polynom) given current sampled t and d
            for di in dspace:
                lat = QuinticPolynomial(c_d[0], c_d[1], c_d[2], di, 0., 0., ti)

                ## sample exact points along the quintic polynom
                d_pos, d_vel, d_acc, d_jrk = [], [], [], []
                for t in t_ticks:
                    d_pos.append(lat.calc_point(t))
                    d_vel.append(lat.calc_first_derivative(t))
                    d_acc.append(lat.calc_second_derivative(t))
                    d_jrk.append(lat.calc_third_derivative(t))
                dcost = self.kj * sum(np.power(d_jrk, 2)) + self.kt * ti + self.kd * (d_pos[-1] ** 2)

                lat_plans.append({"pos": d_pos, "vel": d_vel, "acc": d_acc, "cost": dcost})

            ## generate longitudinal plan (quartic polynom) given current sampled t and s
            for si in sspace:
                lon = QuarticPolynomial(c_s[0], c_s[1], c_s[2], t_s[0] + si, t_s[1], ti)

                ## sample exact points along the quintic polynom
                s_pos, s_vel, s_acc, s_jrk = [], [], [], []
                for t in t_ticks:
                    s_pos.append(lon.calc_point(t))
                    s_vel.append(lon.calc_first_derivative(t))
                    s_acc.append(lon.calc_second_derivative(t))
                    s_jrk.append(lon.calc_third_derivative(t))
                scost = self.kj * sum(np.power(s_jrk, 2)) + self.kt * ti + self.kd * (si ** 2)

                lon_plans.append({"pos": s_pos, "vel": s_vel, "acc": s_acc, "cost": scost})
            
            ## combination of each lateral and longitudinal plans
            for lat_plan in lat_plans:
                for lon_plan in lon_plans:
                    paths["path"].append((
                        lat_plan["pos"], lat_plan["vel"], lat_plan["acc"],
                        lon_plan["pos"], lon_plan["vel"], lon_plan["acc"]
                    ))
                    paths["cost"].append(self.k_lat * lat_plan["cost"] + self.k_lon * lon_plan["cost"])
        
        return paths
    

    def LEFT_LANE_CHANGE(self,
        c_d: Tuple[float, float, float],
        c_s: Tuple[float, float, float],
        t_s: Tuple[float, float],
        ds: float,
        n_ds: int) -> Dict[str, List[Any]]:
        """Generate candidate plans in the frenet frame for distance keeping with constant time gap.

        This planning method assumes the leading object has zero jerk and is used for velocity keeping
        behavior. See Sec V.B of the paper.

        Args:
            c_d (Tuple[float, float, float]): Current (d, d_dot, d_ddot).
            c_s (Tuple[float, float, float]): Current (s, s_dot, s_ddot).
            t_s (Tuple[float, float]): Target (s_dot, s_ddot).
            ds (float): Deviation from S_dot for search space.
            n_ds (int): Number of ds to be sampled in + and - direction.

        Returns:
            Dist[str, List[Any]]: Dictionary with a paired keys=("cost", "path").
        """
        tspace = 2 + np.arange(-self.n_dt * self.dt, (self.n_dt + 1) * self.dt - 1e-6, self.dt)
        dspace = np.arange(0, 3 * self.dd + 1e-6, self.dd)
        sspace = np.arange(-n_ds * ds, (n_ds + 1) * ds - 1e-6, ds)

        paths = {"cost": [], "path": []}

        ## iterate through ts
        for ti in tspace:
            t_ticks = [t for t in np.arange(0, ti, self.dt)]
            lat_plans = []
            lon_plans = []

            ## generate lateral plan (quintic polynom) given current sampled t and d
            for di in dspace:
                lat = QuinticPolynomial(c_d[0], c_d[1], c_d[2], di, 0., 0., ti)

                ## sample exact points along the quintic polynom
                d_pos, d_vel, d_acc, d_jrk = [], [], [], []
                for t in t_ticks:
                    d_pos.append(lat.calc_point(t))
                    d_vel.append(lat.calc_first_derivative(t))
                    d_acc.append(lat.calc_second_derivative(t))
                    d_jrk.append(lat.calc_third_derivative(t))
                dcost = self.kj * sum(np.power(d_jrk, 2)) + self.kt * ti + self.kd * (d_pos[-1] ** 2)

                lat_plans.append({"pos": d_pos, "vel": d_vel, "acc": d_acc, "cost": dcost})

            ## generate longitudinal plan (quartic polynom) given current sampled t and s
            for si in sspace:
                lon = QuarticPolynomial(c_s[0], c_s[1], c_s[2], t_s[0] + si, t_s[1], ti)

                ## sample exact points along the quintic polynom
                s_pos, s_vel, s_acc, s_jrk = [], [], [], []
                for t in t_ticks:
                    s_pos.append(lon.calc_point(t))
                    s_vel.append(lon.calc_first_derivative(t))
                    s_acc.append(lon.calc_second_derivative(t))
                    s_jrk.append(lon.calc_third_derivative(t))
                scost = self.kj * sum(np.power(s_jrk, 2)) + self.kt * ti + self.kd * (si ** 2)

                lon_plans.append({"pos": s_pos, "vel": s_vel, "acc": s_acc, "cost": scost})
            
            ## combination of each lateral and longitudinal plans
            for lat_plan in lat_plans:
                for lon_plan in lon_plans:
                    paths["path"].append((
                        lat_plan["pos"], lat_plan["vel"], lat_plan["acc"],
                        lon_plan["pos"], lon_plan["vel"], lon_plan["acc"]
                    ))
                    paths["cost"].append(self.k_lat * lat_plan["cost"] + self.k_lon * lon_plan["cost"])
        
        return paths