#!/usr/bin/env python3
"""
vis/traj.py  –  Trajectory visualisation for kf_vio_pnp

Reads the three CSV logs produced by kf_node.py / kf_node_cpp:
  - vio_traj_*.csv        : raw VIO positions, velocities, quaternion (world frame)
  - kf_traj_*.csv         : KF-fused positions, velocities, quaternion
  - pnp_detections_*.csv  : PnP-derived quadrotor positions in world frame

Plots
  Figure 1 ─ left  : 2-D XY trajectory  (VIO + KF, coloured by speed; PnP as crosses)
           ─ right : 3-D trajectory     (same components)
  Figure 2  : Euler-angle comparison  (VIO vs KF fused)  – roll / pitch / yaw vs time
"""

import argparse
import glob
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3d projection


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _latest_csv(log_dir: str, prefix: str) -> str:
    """Return the most-recently modified CSV whose name starts with *prefix*."""
    pattern = os.path.join(log_dir, f'{prefix}_*.csv')
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(
            f"No CSV matching '{pattern}' found in '{log_dir}'"
        )
    return files[-1]


def load_traj(path: str) -> pd.DataFrame:
    """Load a trajectory CSV.

    Required columns : timestamp, px, py, pz, vx, vy, vz
    Optional columns : qw, qx, qy, qz  (present in logs from kf_node_cpp)
    """
    df = pd.read_csv(path)
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
    # Add quaternion columns with identity defaults when absent (old log format)
    for col, default in [('qw', 1.0), ('qx', 0.0), ('qy', 0.0), ('qz', 0.0)]:
        if col not in df.columns:
            df[col] = default
    return df


def load_pnp(path: str) -> pd.DataFrame:
    """Load a PnP detection CSV (columns: timestamp, px, py, pz)."""
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Quaternion → Euler angles (ZYX / aerospace convention)
# ---------------------------------------------------------------------------

def quat_to_euler_deg(qw: np.ndarray,
                      qx: np.ndarray,
                      qy: np.ndarray,
                      qz: np.ndarray):
    """Convert quaternion arrays [qw,qx,qy,qz] to roll/pitch/yaw in degrees.

    Convention: ZYX extrinsic (yaw → pitch → roll), i.e. intrinsic XYZ.
    Each input may be a scalar or 1-D numpy array.
    Returns (roll_deg, pitch_deg, yaw_deg) all in degrees.
    """
    # Roll (rotation about X)
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.degrees(np.arctan2(sinr, cosr))

    # Pitch (rotation about Y)  – clamp to avoid arcsin domain errors
    sinp = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))

    # Yaw (rotation about Z)
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.degrees(np.arctan2(siny, cosy))

    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _speed_colours(speed: np.ndarray, cmap_name: str = 'coolwarm') -> np.ndarray:
    """Normalise *speed* to [0, 1] and map through *cmap_name* (blue→red)."""
    cmap = plt.get_cmap(cmap_name)
    vmin, vmax = speed.min(), speed.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))
    return cmap(norm(speed)), norm, cmap


def _coloured_line_2d(ax, x, y, colours):
    """Draw a polyline on *ax* with per-segment colours from *colours* array."""
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], color=colours[i], linewidth=1.2, alpha=0.85)


def _coloured_line_3d(ax, x, y, z, colours):
    """Draw a 3-D polyline with per-segment colours."""
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2],
                color=colours[i], linewidth=1.2, alpha=0.85)


# ---------------------------------------------------------------------------
# Euler-angle comparison plot
# ---------------------------------------------------------------------------

def plot_euler(axes, vio: pd.DataFrame, kf: pd.DataFrame):
    """Plot roll / pitch / yaw comparison (VIO vs KF fused) against time.

    *axes* must be an iterable of 3 Axes objects (one per angle).
    Time axis is seconds elapsed from the start of the VIO log.
    Angles are unwrapped to avoid ±180° discontinuities.
    """
    # Common zero reference so both series start at t=0 s
    t0 = min(vio['timestamp'].iloc[0], kf['timestamp'].iloc[0])

    vio_x = np.arange(len(vio))
    kf_x  = np.arange(len(kf))

    vio_roll_r, vio_pitch_r, vio_yaw_r = quat_to_euler_deg(
        vio['qw'].values, vio['qx'].values, vio['qy'].values, vio['qz'].values)
    kf_roll_r,  kf_pitch_r,  kf_yaw_r  = quat_to_euler_deg(
        kf['qw'].values,  kf['qx'].values,  kf['qy'].values,  kf['qz'].values)

    # Unwrap each angle series to remove ±180° wrap-around discontinuities
    def unwrap_deg(arr):
        return np.degrees(np.unwrap(np.radians(arr)))

    vio_data = [unwrap_deg(vio_roll_r),  unwrap_deg(vio_pitch_r),  unwrap_deg(vio_yaw_r)]
    kf_data  = [unwrap_deg(kf_roll_r),   unwrap_deg(kf_pitch_r),   unwrap_deg(kf_yaw_r)]
    titles   = ['Roll', 'Pitch', 'Yaw']

    for i, (ax, title, vd, kd) in enumerate(zip(axes, titles, vio_data, kf_data)):
        ax.plot(vio_x, vd, color='tab:blue', lw=1.2, alpha=0.85, label='VIO (raw)')
        ax.plot(kf_x,  kd, color='tab:red',  lw=1.2, alpha=0.85, label='KF (fused)')
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('rad')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        # Only label the x-axis on the bottom subplot
        if i == len(titles) - 1:
            ax.set_xlabel('Sample index')


# ---------------------------------------------------------------------------
# Core plot functions
# ---------------------------------------------------------------------------

def plot_2d(ax, vio: pd.DataFrame, kf: pd.DataFrame,
            pnp: Optional[pd.DataFrame]):
    """2-D XY trajectory plot.  VIO = Blues, KF = Reds, both coloured by speed."""
    speed_all = np.concatenate([vio['speed'].values, kf['speed'].values])
    shared_norm = mcolors.Normalize(vmin=speed_all.min(),
                                    vmax=max(speed_all.max(), speed_all.min() + 1e-6))

    # VIO trajectory – Blues colourmap (dark blue = fast)
    vio_cmap = plt.get_cmap('Blues')
    v_cols = vio_cmap(shared_norm(vio['speed'].values))
    _coloured_line_2d(ax, vio['px'].values, vio['py'].values, v_cols)
    ax.plot([], [], color=vio_cmap(0.8), lw=2, label='VIO (raw)')  # legend proxy

    # KF trajectory – Reds colourmap (dark red = fast)
    kf_cmap = plt.get_cmap('Reds')
    kf_cols = kf_cmap(shared_norm(kf['speed'].values))
    _coloured_line_2d(ax, kf['px'].values, kf['py'].values, kf_cols)
    ax.plot([], [], color=kf_cmap(0.8), lw=2, label='KF (fused)')  # legend proxy

    # PnP detections as crosses
    if pnp is not None and not pnp.empty:
        ax.scatter(pnp['px'].values, pnp['py'].values,
                   marker='x', s=60, c='black', linewidths=1.5,
                   zorder=5, label='PnP detections')

    # Start / end markers
    ax.plot(vio['px'].iloc[0], vio['py'].iloc[0],
            'go', ms=8, zorder=6, label='Start')
    ax.plot(kf['px'].iloc[-1], kf['py'].iloc[-1],
            'rs', ms=8, zorder=6, label='End (KF)')

    # Shared colourbar for speed
    sm = plt.cm.ScalarMappable(cmap='Greys', norm=shared_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Speed (m/s) – dark = fast', fontsize=10)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('2-D XY Trajectory  (blue = VIO, red = KF)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def plot_3d(ax, vio: pd.DataFrame, kf: pd.DataFrame,
            pnp: Optional[pd.DataFrame]):
    """3-D trajectory plot.  VIO = Blues, KF = Reds, both coloured by speed."""
    speed_all = np.concatenate([vio['speed'].values, kf['speed'].values])
    shared_norm = mcolors.Normalize(vmin=speed_all.min(),
                                    vmax=max(speed_all.max(), speed_all.min() + 1e-6))

    vio_cmap = plt.get_cmap('Blues')
    v_cols = vio_cmap(shared_norm(vio['speed'].values))
    _coloured_line_3d(ax,
                      vio['px'].values, vio['py'].values, vio['pz'].values,
                      v_cols)
    ax.plot([], [], [], color=vio_cmap(0.8), lw=2, label='VIO (raw)')

    kf_cmap = plt.get_cmap('Reds')
    kf_cols = kf_cmap(shared_norm(kf['speed'].values))
    _coloured_line_3d(ax,
                      kf['px'].values, kf['py'].values, kf['pz'].values,
                      kf_cols)
    ax.plot([], [], [], color=kf_cmap(0.8), lw=2, label='KF (fused)')

    if pnp is not None and not pnp.empty:
        ax.scatter(pnp['px'].values, pnp['py'].values, pnp['pz'].values,
                   marker='x', s=60, c='black', linewidths=1.5,
                   depthshade=False, zorder=5, label='PnP detections')

    ax.scatter(*[vio[c].iloc[0] for c in ('px', 'py', 'pz')],
               color='green', s=60, zorder=6, label='Start')
    ax.scatter(*[kf[c].iloc[-1] for c in ('px', 'py', 'pz')],
               color='red', marker='s', s=60, zorder=6, label='End (KF)')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3-D Trajectory')
    ax.legend(fontsize=9)


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot VIO / KF / PnP trajectory from kf_node.py logs.')
    parser.add_argument(
        '--log_dir',
        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'logs'),
        help='Directory containing the CSV log files (default: data/logs/)')
    parser.add_argument(
        '--vio',  default=None, help='Path to vio_traj CSV (overrides auto-detect)')
    parser.add_argument(
        '--kf',   default=None, help='Path to kf_traj CSV (overrides auto-detect)')
    parser.add_argument(
        '--pnp',  default=None, help='Path to pnp_detections CSV (overrides auto-detect)')
    parser.add_argument(
        '--cmap', default='coolwarm',
        help='Matplotlib colourmap for speed (default: coolwarm)')
    args = parser.parse_args()

    log_dir = os.path.abspath(args.log_dir)

    # Resolve CSV paths
    vio_path = args.vio or _latest_csv(log_dir, 'vio_traj')
    kf_path  = args.kf  or _latest_csv(log_dir, 'kf_traj')

    try:
        pnp_path = args.pnp or _latest_csv(log_dir, 'pnp_detections')
        pnp = load_pnp(pnp_path)
        print(f"PnP detections : {pnp_path}  ({len(pnp)} points)")
    except FileNotFoundError:
        pnp = None
        print("No pnp_detections CSV found – skipping PnP overlay.")

    vio = load_traj(vio_path)
    kf  = load_traj(kf_path)

    print(f"VIO trajectory : {vio_path}  ({len(vio)} samples)")
    print(f"KF  trajectory : {kf_path}   ({len(kf)} samples)")

    # -----------------------------------------------------------------------
    # Figure 1: 2-D left, 3-D right
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 7))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')

    plot_2d(ax2d, vio, kf, pnp)
    plot_3d(ax3d, vio, kf, pnp)

    plt.suptitle('kf_vio_pnp Trajectory Visualisation', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # -----------------------------------------------------------------------
    # Figure 2: Euler-angle comparison
    # -----------------------------------------------------------------------
    fig2, euler_axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    plot_euler(euler_axes, vio, kf)
    fig2.suptitle('VIO vs KF Fused Orientation (Euler angles, ZYX convention)',
                  fontsize=13, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.96])  # leave room for suptitle

    plt.show()


if __name__ == '__main__':
    main()
