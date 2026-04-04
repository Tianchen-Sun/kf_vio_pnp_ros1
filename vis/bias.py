import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

def plot_bias_visualization(csv_file_path):
    """
    Visualize bias data with 2D time series and 3D trajectory
    
    Args:
        csv_file_path: Path to the bias data CSV file
    """
    # Read CSV file
    df = pd.read_csv(csv_file_path)
    
    # Extract data
    timestamps = df['timestamp'].values
    bias_x = df['bias_x'].values
    bias_y = df['bias_y'].values
    bias_z = df['bias_z'].values
    
    # Normalize timestamps to start from 0
    t_normalized = timestamps - timestamps[0]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 2D Time series plots (3 subplots for each axis)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t_normalized, bias_x, 'r-', linewidth=1.5, label='Bias X')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Bias X (m)', fontsize=11)
    ax1.set_title('VIO Position Bias - X Axis', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(t_normalized, bias_y, 'g-', linewidth=1.5, label='Bias Y')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Bias Y (m)', fontsize=11)
    ax2.set_title('VIO Position Bias - Y Axis', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3D trajectory plot
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    scatter = ax3.scatter(bias_x, bias_y, bias_z, c=t_normalized, cmap='viridis', 
                         s=20, alpha=0.6, edgecolors='none')
    ax3.plot(bias_x, bias_y, bias_z, 'k-', alpha=0.2, linewidth=0.5)
    ax3.set_xlabel('Bias X (m)', fontsize=10)
    ax3.set_ylabel('Bias Y (m)', fontsize=10)
    ax3.set_zlabel('Bias Z (m)', fontsize=10)
    ax3.set_title('3D Bias Trajectory', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax3, pad=0.1, shrink=0.8)
    cbar.set_label('Time (s)', fontsize=10)
    
    # Z-axis time series
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(t_normalized, bias_z, 'b-', linewidth=1.5, label='Bias Z')
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Bias Z (m)', fontsize=11)
    ax4.set_title('VIO Position Bias - Z Axis', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Print statistics
    print("\n" + "="*60)
    print("BIAS STATISTICS")
    print("="*60)
    print(f"Duration: {t_normalized[-1]:.2f} seconds")
    print(f"Number of samples: {len(df)}")
    print("\nBias X (m):")
    print(f"  Mean: {np.mean(bias_x):.6f}")
    print(f"  Std:  {np.std(bias_x):.6f}")
    print(f"  Min:  {np.min(bias_x):.6f}")
    print(f"  Max:  {np.max(bias_x):.6f}")
    print("\nBias Y (m):")
    print(f"  Mean: {np.mean(bias_y):.6f}")
    print(f"  Std:  {np.std(bias_y):.6f}")
    print(f"  Min:  {np.min(bias_y):.6f}")
    print(f"  Max:  {np.max(bias_y):.6f}")
    print("\nBias Z (m):")
    print(f"  Mean: {np.mean(bias_z):.6f}")
    print(f"  Std:  {np.std(bias_z):.6f}")
    print(f"  Min:  {np.min(bias_z):.6f}")
    print(f"  Max:  {np.max(bias_z):.6f}")
    print("="*60 + "\n")
    
    plt.show()

# Usage example:
if __name__ == "__main__":
    csv_path = "data/logs/bias_data_20260404_183821.csv"  # Update with your actual CSV file path
    plot_bias_visualization(csv_path)