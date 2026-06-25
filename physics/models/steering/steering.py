import numpy as np
from scipy.optimize import brentq, fminbound

def sw2iso(sw_coords: np.ndarray | list[float]) -> np.ndarray:
    """Converts SolidWorks coordinates to ISO 8855 coordinates."""
    return np.array([sw_coords[2], sw_coords[0], sw_coords[1]])

def get_pt_on_axis(p1: np.ndarray, p2: np.ndarray, h: float) -> np.ndarray:
    """Finds the 3D point on the line through p1 and p2 where z == h."""
    z1 = p1[2]
    z2 = p2[2]
    if abs(z1 - z2) < 1e-12:
        raise ValueError("get_pt_on_axis() requires points with unique z")
    t = (h - z1) / (z2 - z1)
    q = p1 + t * (p2 - p1)
    q[2] = h
    return q

def get_revolve_path(axis_fun, point: np.ndarray):
    """Gets parametric function defining circular revolution path around an axis."""
    axis_pt = axis_fun(0.0)
    axis_dir = axis_fun(1.0) - axis_pt
    v = point - axis_pt
    k = axis_dir / np.linalg.norm(axis_dir)
    
    def path_fun(t: float) -> np.ndarray:
        # Rodrigues' rotation formula
        return (v * np.cos(t) + 
                np.cross(k, v) * np.sin(t) + 
                k * np.dot(k, v) * (1.0 - np.cos(t)) + 
                axis_pt)
    return path_fun

def rot_about_axis(axis_fun, point: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotates a point about an axis defined by a parametric function axis_fun."""
    axis_pt = axis_fun(0.0)
    axis_dir = axis_fun(1.0) - axis_pt
    v = point - axis_pt
    k = axis_dir / np.linalg.norm(axis_dir)
    v_rot = (v * np.cos(angle_rad) + 
             np.cross(k, v) * np.sin(angle_rad) + 
             k * np.dot(k, v) * (1.0 - np.cos(angle_rad)))
    return v_rot + axis_pt

def ring_point_min_z(c: np.ndarray, v: np.ndarray, R: float) -> np.ndarray:
    """Computes the point on a ring around axis v with minimal z."""
    v_hat = v / np.linalg.norm(v)
    u = np.cross(v_hat, np.array([0.0, 0.0, 1.0]))
    u = u / np.linalg.norm(u)
    w = np.cross(v_hat, u)
    
    theta = np.arctan2(-w[2], -u[2])
    return c + R * (np.cos(theta) * u + np.sin(theta) * w)

def get_delta(wheel_center_pt: np.ndarray, spindle_pt: np.ndarray) -> float:
    """Calculates the steer angle of a wheel based on spindle orientation."""
    spindle_vec = wheel_center_pt - spindle_pt
    up = np.array([0.0, 0.0, 1.0])
    fwd = np.array([1.0, 0.0, 0.0])
    
    heading = np.cross(spindle_vec, up)
    norm_val = np.linalg.norm(heading)
    if norm_val < 1e-12:
        return 0.0
    heading = heading / norm_val
    if heading[0] < 0:
        heading = -heading
        
    dot_val = np.clip(np.dot(heading, fwd), -1.0, 1.0)
    angle_mag = np.arccos(dot_val)
    if angle_mag <= np.finfo(float).eps:
        return 0.0
    else:
        angle_dir = np.sign(np.dot(np.cross(fwd, heading), up))
        return angle_dir * angle_mag


class SteeringModel:
    """
    Steering Linkage Kinematics Solver.
    Computes Left and Right wheel steer angles from Yoke Position input.
    """
    def __init__(
        self,
        wheelbase: float = 2600.0,
        trackwidth: float = 1270.0,
        cfactor: float = 4.71 * 25.4,      # ~119.634 mm/rev
        wheel_radius: float = 283.0,
        sw_ubj: list[float] = [-185.0, 610.0, -15.04],
        sw_lbj: list[float] = [-80.0, 175.0, 4.96],
        er_z: float = 695.58,
        sa_z0: float = 650.0,
        setback: float = 520.0,
        er_connection_len: float = 30.0,
        sa_vec: list[float] = [-75.0, 35.0]
    ):
        self.wheelbase = wheelbase
        self.trackwidth = trackwidth
        self.cfactor = cfactor
        self.wheel_radius = wheel_radius
        self.sw_ubj = sw_ubj
        self.sw_lbj = sw_lbj
        self.er_z = er_z
        self.sa_z0 = sa_z0
        self.setback = setback
        self.er_connection_len = er_connection_len
        self.sa_vec = sa_vec
        
        self._compute_static_nodes()

    def _compute_static_nodes(self):
        # Convert SolidWorks coordinates to ISO 8855
        tp_ubj = sw2iso(self.sw_ubj)
        tp_lbj = sw2iso(self.sw_lbj)
        
        # Tire Patch Reference Frame Points
        tp_kp = get_pt_on_axis(tp_ubj, tp_lbj, self.wheel_radius)
        tp_spindle = np.array([0.0, tp_kp[1], tp_kp[2]])
        tp_kp_sa = get_pt_on_axis(tp_ubj, tp_lbj, self.sa_z0)
        tp_sa_tr = tp_kp_sa + np.array([self.sa_vec[0], self.sa_vec[1], 0.0])
        
        # Static Nodes (Port/Left Side)
        self.er_axis_port = lambda dy: np.array([
            self.wheelbase - self.setback,
            0.5 * self.er_connection_len + dy,
            self.er_z
        ])
        
        s_nodes_port = {}
        s_nodes_port['TP'] = np.array([self.wheelbase, 0.5 * self.trackwidth, 0.0])
        s_nodes_port['WC'] = s_nodes_port['TP'] + np.array([0.0, 0.0, self.wheel_radius])
        s_nodes_port['UBJ'] = s_nodes_port['TP'] + tp_ubj
        s_nodes_port['LBJ'] = s_nodes_port['TP'] + tp_lbj
        s_nodes_port['KP'] = s_nodes_port['TP'] + tp_kp
        s_nodes_port['spindle'] = s_nodes_port['TP'] + tp_spindle
        s_nodes_port['KP_SA'] = s_nodes_port['TP'] + tp_kp_sa
        s_nodes_port['SA_TR'] = s_nodes_port['TP'] + tp_sa_tr
        s_nodes_port['ER_TR'] = self.er_axis_port(0.0)
        
        # Static Nodes (Starboard/Right Side) - Mirrored Y components
        self.er_axis_starboard = lambda dy: self.er_axis_port(dy) - np.array([0.0, self.er_connection_len, 0.0])
        
        s_nodes_starboard = {}
        for key, val in s_nodes_port.items():
            s_nodes_starboard[key] = np.array([val[0], -val[1], val[2]])
        s_nodes_starboard['ER_TR'] = self.er_axis_starboard(0.0)
        
        self.s_nodes_port = s_nodes_port
        self.s_nodes_starboard = s_nodes_starboard

    def _solve_steering_linkage(self, s_nodes, er_axis_fun, rack_shift, is_port):
        # Calculate tie rod length
        tie_rod_len = np.linalg.norm(s_nodes['ER_TR'] - s_nodes['SA_TR'])
        
        # KPaxisFun
        kp_axis_dir = (s_nodes['UBJ'] - s_nodes['LBJ']) / np.linalg.norm(s_nodes['UBJ'] - s_nodes['LBJ'])
        kp_axis_fun = lambda t: s_nodes['LBJ'] + t * kp_axis_dir
        
        # SApathFun
        sa_path_fun = get_revolve_path(kp_axis_fun, s_nodes['SA_TR'])
        
        # Dynamic ER_TR
        er_tr = er_axis_fun(rack_shift)
        
        # Slack function
        def slack_fun(theta):
            path_pt = sa_path_fun(is_port * theta)
            return tie_rod_len - np.linalg.norm(path_pt - er_tr)
            
        val = np.sign(rack_shift) * is_port
        
        if abs(val) < 1e-12:  # static position
            d_nodes = s_nodes.copy()
            d_nodes['TPrigid'] = s_nodes['TP'].copy()
            spin = 0.0
            return d_nodes, spin
            
        elif val > 0:  # Case 1: steering arm swings CW (rack approaching)
            theta_negative = fminbound(slack_fun, -np.pi, 0.0, xtol=1e-6)
            path_domain = (theta_negative, 0.0)
            
        else:  # Case -1: steering arm swings CCW (rack retreating)
            theta_positive = fminbound(lambda t: -slack_fun(t), 0.0, np.pi, xtol=1e-6)
            if slack_fun(theta_positive) < 0.0:
                raise ValueError(
                    f"No upright orientation satisfies {rack_shift:.4f} mm rack position! "
                    "Linkage overextended!"
                )
            path_domain = (0.0, theta_positive)
            
        # Solve for upright rotation angle
        try:
            theta_root = brentq(slack_fun, path_domain[0], path_domain[1], xtol=1e-12)
        except ValueError as e:
            raise ValueError(
                f"No upright orientation satisfies {rack_shift:.4f} mm rack position! "
                "Linkage is disconnected or overextended."
            ) from e
            
        spin = is_port * theta_root
        
        # Compute dynamic node positions
        d_nodes = {}
        d_nodes['SA_TR'] = sa_path_fun(spin)
        d_nodes['TPrigid'] = rot_about_axis(kp_axis_fun, s_nodes['TP'], spin)
        d_nodes['WC'] = rot_about_axis(kp_axis_fun, s_nodes['WC'], spin)
        d_nodes['spindle'] = rot_about_axis(kp_axis_fun, s_nodes['spindle'], spin)
        
        # Static nodes unchanged
        d_nodes['UBJ'] = s_nodes['UBJ']
        d_nodes['LBJ'] = s_nodes['LBJ']
        d_nodes['KP'] = s_nodes['KP']
        d_nodes['KP_SA'] = s_nodes['KP_SA']
        d_nodes['ER_TR'] = er_tr
        
        # Find tire contact patch (lowest point on wheel)
        d_nodes['TP'] = ring_point_min_z(d_nodes['WC'], d_nodes['WC'] - d_nodes['spindle'], self.wheel_radius)
        
        return d_nodes, spin

    def calculate_steer_angles(self, yoke_deg: float) -> tuple[float, float]:
        """
        Calculates the left and right steer angles in degrees for a given yoke angle in degrees.
        Positive yoke angle is a left turn.
        
        Returns:
            (delta_left, delta_right) in degrees
        """
        # Convert yoke angle (degrees) to rack position (mm)
        rack_shift = - (self.cfactor * yoke_deg / 360.0)
        
        # Port (Left) Side Linkage Kinematics
        nodes_port, _ = self._solve_steering_linkage(
            self.s_nodes_port, self.er_axis_port, rack_shift, is_port=1
        )
        
        # Starboard (Right) Side Linkage Kinematics
        nodes_starboard, _ = self._solve_steering_linkage(
            self.s_nodes_starboard, self.er_axis_starboard, rack_shift, is_port=-1
        )
        
        # Calculate steer angles using WC and spindle
        delta_l_rad = get_delta(nodes_port['WC'], nodes_port['spindle'])
        delta_r_rad = get_delta(nodes_starboard['WC'], nodes_starboard['spindle'])
        
        return np.degrees(delta_l_rad), np.degrees(delta_r_rad)
