import numpy as np


class Transform:
    """
    Transform class for coordinate transformation between PnP and VIO frames.
    Handles rotation (yaw only) and translation.
    VIO frame is defined relative to PnP frame.
    """
    
    def __init__(self, vio_yaw_rel_pnp=0.0, vio_translation_rel_pnp=None):
        """
        Initialize transformation parameters.
        VIO frame pose relative to PnP frame.
        """
        self.vio_yaw_rel_pnp = vio_yaw_rel_pnp
        if vio_translation_rel_pnp is not None:
            self.vio_translation_rel_pnp = np.array(vio_translation_rel_pnp)
        else:
            self.vio_translation_rel_pnp = np.array([0.0, 0.0, 0.0])
        
        self._compute_rotation_matrices()
    
    def _compute_rotation_matrices(self):
        """Compute rotation matrices for yaw angle"""
        cos_y = np.cos(self.vio_yaw_rel_pnp)
        sin_y = np.sin(self.vio_yaw_rel_pnp)
        self.R_vio_to_pnp = np.array([
            [cos_y, -sin_y, 0],
            [sin_y, cos_y, 0],
            [0, 0, 1]
        ], dtype=float)
        
        self.R_pnp_to_vio = self.R_vio_to_pnp.T
    
    def set_vio_frame(self, yaw, translation=None):
        """Update VIO frame parameters relative to PnP"""
        self.vio_yaw_rel_pnp = yaw
        if translation is not None:
            self.vio_translation_rel_pnp = np.array(translation)
        self._compute_rotation_matrices()
    
    def pnp_to_vio(self, pnp_position):
        """Transform position from PnP frame to VIO frame."""
        pnp_pos = np.array(pnp_position, dtype=float)
        pos_relative = pnp_pos - self.vio_translation_rel_pnp
        pos_vio = self.R_pnp_to_vio @ pos_relative
        return pos_vio
    
    def vio_to_pnp(self, vio_position):
        """Transform position from VIO frame to PnP frame."""
        vio_pos = np.array(vio_position, dtype=float)
        pos_pnp_relative = self.R_vio_to_pnp @ vio_pos
        pos_pnp = pos_pnp_relative + self.vio_translation_rel_pnp
        return pos_pnp

    def vio_vector_to_pnp(self, vio_vector):
        """Transform a free vector (e.g. velocity) from VIO to PnP frame."""
        vec_vio = np.array(vio_vector, dtype=float)
        return self.R_vio_to_pnp @ vec_vio
    
    def yaw_vio_to_pnp(self, yaw_vio):
        """Transform yaw angle from VIO frame to PnP frame"""
        return yaw_vio + self.vio_yaw_rel_pnp

    def quaternion_multiply(self, q1, q2):
        """Hamilton product of two quaternions in ROS order [x, y, z, w]."""
        x1, y1, z1, w1 = np.array(q1, dtype=float)
        x2, y2, z2, w2 = np.array(q2, dtype=float)
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ], dtype=float)

    def quaternion_vio_to_pnp(self, q_vio):
        """
        Transform orientation from VIO frame to PnP frame using the configured
        yaw offset between frames.
        """
        q_offset = self.yaw_to_quaternion(self.vio_yaw_rel_pnp)
        q_pnp = self.quaternion_multiply(q_offset, q_vio)
        q_pnp = q_pnp / np.linalg.norm(q_pnp)
        return q_pnp

    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)
        
        Args:
            roll, pitch, yaw: Euler angles in radians
            
        Returns:
            Quaternion as [x, y, z, w] (ROS standard format)
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return np.array([qx, qy, qz, qw])
    
    def quaternion_to_euler(self, qx, qy, qz, qw):
        """
        Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw)
        
        Args:
            qx, qy, qz, qw: Quaternion components
            
        Returns:
            Euler angles as (roll, pitch, yaw) in radians
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        sinp = np.clip(sinp, -1, 1)
        pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def yaw_to_quaternion(self, yaw):
        """
        Convert yaw angle to quaternion (assuming roll=0, pitch=0)
        
        Args:
            yaw: Yaw angle in radians
            
        Returns:
            Quaternion as [x, y, z, w] (ROS standard format)
        """
        half_yaw = yaw / 2.0
        return np.array([
            0.0,                    # qx
            0.0,                    # qy
            np.sin(half_yaw),       # qz
            np.cos(half_yaw)        # qw
        ])
    
    def quaternion_to_yaw(self, q):
        """
        Extract yaw angle from quaternion
        
        Args:
            q: Quaternion as [x, y, z, w]
            
        Returns:
            Yaw angle in radians
        """
        qx, qy, qz, qw = q
        # Extract yaw from quaternion
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw


class ENUtoNEDTransform:
    """
    Transform class for coordinate transformation between ENU and NED frames.
    ENU: East-North-Up coordinate system (Z points up)
    NED: North-East-Down coordinate system (Z points down)
    """
    
    def __init__(self):
        """Initialize ENU to NED transformation."""
        # Transformation matrix from ENU to NED
        # ENU: [East, North, Up] -> NED: [North, East, -Up]
        self.T_enu_to_ned = np.array([
            [0, 1, 0],   # NED_x = ENU_y (North)
            [1, 0, 0],   # NED_y = ENU_x (East)
            [0, 0, -1]   # NED_z = -ENU_z (Down)
        ], dtype=float)
        
        self.T_ned_to_enu = self.T_enu_to_ned.T
    
    def enu_to_ned_position(self, enu_position):
        """
        Transform position from ENU frame to NED frame.
        
        Args:
            enu_position: Position as [east, north, up]
            
        Returns:
            Position in NED frame as [north, east, down]
        """
        enu_pos = np.array(enu_position, dtype=float)
        ned_pos = self.T_enu_to_ned @ enu_pos
        return ned_pos
    
    def ned_to_enu_position(self, ned_position):
        """
        Transform position from NED frame to ENU frame.
        
        Args:
            ned_position: Position as [north, east, down]
            
        Returns:
            Position in ENU frame as [east, north, up]
        """
        ned_pos = np.array(ned_position, dtype=float)
        enu_pos = self.T_ned_to_enu @ ned_pos
        return enu_pos
    
    def enu_to_ned_yaw(self, enu_yaw):
        """
        Transform yaw angle from ENU frame to NED frame.
        
        Args:
            enu_yaw: Yaw angle in ENU frame (radians)
            
        Returns:
            Yaw angle in NED frame (radians)
        """
        # Yaw is negated because Z-axis points in opposite direction
        return -enu_yaw
    
    def ned_to_enu_yaw(self, ned_yaw):
        """
        Transform yaw angle from NED frame to ENU frame.
        
        Args:
            ned_yaw: Yaw angle in NED frame (radians)
            
        Returns:
            Yaw angle in ENU frame (radians)
        """
        # Yaw is negated because Z-axis points in opposite direction
        return -ned_yaw


def rotation_matrix_yaw(yaw):
    """Create a rotation matrix for yaw angle only."""
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    return np.array([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1]
    ])


def apply_rotation(position, yaw):
    """Apply yaw rotation to a position vector."""
    R = rotation_matrix_yaw(yaw)
    return R @ np.array(position)


def apply_translation(position, translation):
    """Apply translation to a position vector."""
    return np.array(position) + np.array(translation)