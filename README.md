# kalman filter for fusion VIO and PnP detection Results
1. Config the initial Drone position in the world frame.
2. Double check the `scene.csv`, it store all gate poses in the World frame, only `[cx, cy, cz]` used.

3. This node will detect `d2vins/odometry` and `gate_pose/pose` topics automatically and fuse.

# How to run
by default the node is run alongside with `perpare.sh`. you can run mannually via
```
roslaunch kf_vio_pnp kf_vio_pnp.launch
```