#!/usr/bin/env python3
"""
vis/traj.py  –  Trajectory visualisation for kf_vio_pnp

Reads the three CSV logs produced by kf_node.py:
  - vio_traj_*.csv        : raw VIO positions & velocities (in PnP/world frame)
  - kf_traj_*.csv         : KF-fused positions & velocities
  - pnp_detections_*.csv  : PnP-derived quadrotor positions in world frame

Plots
  1. 2-D XY trajectory  (VIO + KF, coloured by speed; PnP as crosses)
  2. 3-D trajectory     (same components)
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
    """Load a trajectory CSV (columns: timestamp, px, py, pz, vx, vy, vz)."""
    df = pd.read_csv(path)
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
    return df


def load_pnp(path: str) -> pd.DataFrame:
    """Load a PnP detection CSV (columns: timestamp, px, py, pz)."""
    return pd.read_csv(path)


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
    # Figure layout: 2-D left, 3-D right
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 7))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')

    plot_2d(ax2d, vio, kf, pnp)
    plot_3d(ax3d, vio, kf, pnp)

    plt.suptitle('kf_vio_pnp Trajectory Visualisation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
