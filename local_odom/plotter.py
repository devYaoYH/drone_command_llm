# plotter.py
"""Manages the 3D Matplotlib plot for Tello visualization."""

import matplotlib
matplotlib.use('TkAgg') # Use Tkinter backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
import config # Import config constants

class PlotManager:
    """Handles the creation, updating, and resetting of the 3D plot."""

    def __init__(self, parent_frame):
        """
        Initializes the 3D plot within the parent Tkinter frame.

        Args:
            parent_frame: The tkinter.Frame where the plot canvas will be placed.
        """
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax3d = self.fig.add_subplot(111, projection='3d')

        # Plot elements
        self.drone_scatter = self.ax3d.scatter([], [], [], c='r', marker='o', s=50, label='Drone (Cmd)')
        self.orientation_line, = self.ax3d.plot([], [], [], 'r-', lw=2, label='Yaw (Cmd)')
        self.trajectory_line, = self.ax3d.plot([], [], [], 'b:', lw=1, label='Trajectory (Cmd)')
        # --- Add VO Trajectory Line ---
        self.vo_trajectory_line, = self.ax3d.plot([], [], [], 'g--', lw=1, label='Trajectory (VO)')

        self._setup_plot_aesthetics()

        # Embed the plot in Tkinter frame
        self.canvas3d = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas3d_widget = self.canvas3d.get_tk_widget()
        self.canvas3d_widget.grid(row=0, column=0, sticky="nsew") # Use grid within the parent frame
        self.canvas3d.draw()

    def _setup_plot_aesthetics(self):
        """Sets labels, title, legend, and initial limits."""
        self.ax3d.set_xlabel('X (cm) - Forward')
        self.ax3d.set_ylabel('Y (cm) - Left')
        self.ax3d.set_zlabel('Z (cm) - Up')
        self.ax3d.set_title('Relative Position')
        self.ax3d.legend(fontsize='small', loc='upper left')
        # self.ax3d.legend(fontsize='small')
        self.ax3d.set_xlim(config.PLOT_INITIAL_LIMITS['xlim'])
        self.ax3d.set_ylim(config.PLOT_INITIAL_LIMITS['ylim'])
        self.ax3d.set_zlim(config.PLOT_INITIAL_LIMITS['zlim'])
        self.ax3d.grid(True)
        # Optional: Equal aspect ratio (can distort view)
        # self.ax3d.set_aspect('equal', adjustable='box')

    def update_plot(self, x_cmd, y_cmd, z_cmd, yaw_deg_cmd, trajectory_cmd, vo_trajectory, vo_enabled):
        """
        Updates the plot elements with new data.

        Args:
            x_cmd (float): Current estimated drone x coordinate.
            y_cmd (float): Current estimated drone y coordinate.
            z_cmd (float): Current estimated drone z coordinate.
            yaw_deg_cmd (float): Current estimated drone yaw in degrees.
            trajectory_cmd (list): List of (x, y, z) tuples representing the command-based path.
            vo_trajectory (list): List of (x, y, z) tuples representing the VO-based path.
            vo_enabled (bool): Whether VO trajectory should be displayed.
        """
        # print(f"  PlotManager.update_plot called. Pos:({x_cmd:.1f},{y_cmd:.1f},{z_cmd:.1f}) Yaw:{yaw_deg_cmd:.1f} CmdTrajLen:{len(trajectory_cmd)} VOTrajLen:{len(vo_trajectory)} VOEnabled:{vo_enabled}")
        if not self.canvas3d or not self.canvas3d_widget.winfo_exists():
            return

        try:
            # --- Update Command-Based Elements ---
            yaw_rad_cmd = math.radians(yaw_deg_cmd)
            self.drone_scatter._offsets3d = ([x_cmd], [y_cmd], [z_cmd])
            vec_len = config.PLOT_ORIENTATION_VECTOR_LENGTH
            end_x = x_cmd + vec_len * math.cos(yaw_rad_cmd)
            end_y = y_cmd + vec_len * math.sin(yaw_rad_cmd)
            self.orientation_line.set_data([x_cmd, end_x], [y_cmd, end_y])
            self.orientation_line.set_3d_properties([z_cmd, z_cmd])

            if trajectory_cmd:
                traj_x, traj_y, traj_z = zip(*trajectory_cmd)
                self.trajectory_line.set_data(traj_x, traj_y)
                self.trajectory_line.set_3d_properties(traj_z)
            else:
                self.trajectory_line.set_data([], [])
                self.trajectory_line.set_3d_properties([])

            # --- Update VO Trajectory ---
            if vo_enabled and vo_trajectory:
                if len(vo_trajectory) > 1:
                    vo_traj_x, vo_traj_y, vo_traj_z = zip(*vo_trajectory)
                    self.vo_trajectory_line.set_data(vo_traj_x, vo_traj_y)
                    self.vo_trajectory_line.set_3d_properties(vo_traj_z)
                    self.vo_trajectory_line.set_visible(True)
                else:
                    self.vo_trajectory_line.set_visible(False)
            else:
                self.vo_trajectory_line.set_visible(False)

            # --- Update axis limits based on all points ---
            all_points = []
            # Add current drone position
            all_points.append((x_cmd, y_cmd, z_cmd)) # <<< ADD THIS LINE
            if trajectory_cmd:
                all_points.extend(trajectory_cmd)
            if vo_enabled and vo_trajectory:
                all_points.extend(vo_trajectory)

            if all_points:
                self._update_axis_limits_from_points(all_points)

            self.canvas3d.draw()

        except Exception as e:
            print(f"Error updating plot: {e}")
            # Don't re-raise, we want to continue even if plotting fails

    def _update_axis_limits_from_points(self, points):
        """Dynamically adjust axis limits based on a list of (x,y,z) points."""
        if not points:
            return

        all_x, all_y, all_z = zip(*points)

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        min_z, max_z = min(all_z), max(all_z)

        pad_x = max(config.PLOT_MIN_PADDING_XYZ[0], abs(max_x - min_x) * config.PLOT_PADDING_FACTOR)
        pad_y = max(config.PLOT_MIN_PADDING_XYZ[1], abs(max_y - min_y) * config.PLOT_PADDING_FACTOR)
        pad_z = max(config.PLOT_MIN_PADDING_XYZ[2], abs(max_z - min_z) * config.PLOT_PADDING_FACTOR)

        # Handle cases where min/max are the same
        if pad_x < 1: pad_x = config.PLOT_MIN_PADDING_XYZ[0]
        if pad_y < 1: pad_y = config.PLOT_MIN_PADDING_XYZ[1]
        if pad_z < 1: pad_z = config.PLOT_MIN_PADDING_XYZ[2]


        self.ax3d.set_xlim(min_x - pad_x, max_x + pad_x)
        self.ax3d.set_ylim(min_y - pad_y, max_y + pad_y)
        self.ax3d.set_zlim(min(0, min_z - pad_z), max_z + pad_z)


    def reset_plot(self):
        """Resets the plot to its initial state (origin)."""
        if not self.canvas3d or not self.canvas3d_widget.winfo_exists():
            return
        try:
            # Reset command-based elements
            self.drone_scatter._offsets3d = ([0], [0], [0])
            vec_len = config.PLOT_ORIENTATION_VECTOR_LENGTH
            self.orientation_line.set_data([0, vec_len], [0, 0]) # Point along +X
            self.orientation_line.set_3d_properties([0, 0])
            self.trajectory_line.set_data([0], [0])
            self.trajectory_line.set_3d_properties([0])

            # Reset VO elements
            self.vo_trajectory_line.set_data([], [])
            self.vo_trajectory_line.set_3d_properties([])
            self.vo_trajectory_line.set_visible(False)

            # Reset limits
            self.ax3d.set_xlim(config.PLOT_INITIAL_LIMITS['xlim'])
            self.ax3d.set_ylim(config.PLOT_INITIAL_LIMITS['ylim'])
            self.ax3d.set_zlim(config.PLOT_INITIAL_LIMITS['zlim'])

            self.canvas3d.draw_idle()
        except Exception as e:
            print(f"Error resetting 3D plot: {e}")

    # Need original _update_axis_limits for backward compatibility if called elsewhere
    def _update_axis_limits(self, trajectory, x, y, z):
         """Original method signature, calls the points-based version."""
         points = list(trajectory) + [(x,y,z)]
         self._update_axis_limits_from_points(points)

    def save_plot(self, filename="tello_plot.png"):
        """
        Saves the current state of the 3D plot to a file.

        Args:
            filename (str): The filename (with path, if needed) to save the plot as.
        """
        try:
            self.fig.savefig(filename, dpi=300)
            print(f"Plot saved to {filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
