# visual_odometry.py
"""
Enhanced Monocular Visual Odometry system for Tello drone using OpenCV
with scale estimation, keyframe selection, and motion filtering.
"""

import cv2
import numpy as np
from collections import deque
import time
from camera_calibration import camera_matrix, dist_coeffs  # Use calibrated values

class VisualOdometry:
    def __init__(self, cam_matrix=camera_matrix, distortion=dist_coeffs, feature_detector_type='ORB'):
        """
        Initializes the Enhanced Visual Odometry system for Tello drone.

        Args:
            cam_matrix: The camera intrinsic matrix (K).
            distortion: The camera distortion coefficients.
            feature_detector_type: 'ORB' or 'SIFT' (SIFT needs opencv-contrib-python).
        """
        self.K = cam_matrix
        self.D = distortion
        self.feature_detector_type = feature_detector_type.upper()

        # Feature Detector and Matcher
        if self.feature_detector_type == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=500,  # Increased features
                                           scaleFactor=1.2,
                                           nlevels=8,
                                           edgeThreshold=31,
                                           firstLevel=0,
                                           WTA_K=2,
                                           scoreType=cv2.ORB_HARRIS_SCORE,
                                           patchSize=31,
                                           fastThreshold=20)
            # Use FLANN for faster matching with ORB
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif self.feature_detector_type == 'SIFT':
            try:
                self.detector = cv2.SIFT_create(nfeatures=500)
                # FLANN for SIFT
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            except AttributeError:
                print("ERROR: SIFT not available. Falling back to ORB. Install opencv-contrib-python if needed.")
                self.feature_detector_type = 'ORB'
                self.detector = cv2.ORB_create(nfeatures=500)
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                    table_number=6,
                                    key_size=12,
                                    multi_probe_level=1)
                search_params = dict(checks=50)
                self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError("Unsupported feature detector type. Choose 'ORB' or 'SIFT'.")

        # State variables
        self.prev_frame_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.reference_keyframe = None  # For keyframe-based tracking
        self.reference_kp = None
        self.reference_desc = None

        # Trajectory tracking
        self.current_R = np.identity(3)  # Current Rotation matrix (world to camera)
        self.current_t = np.zeros((3, 1))  # Current Translation vector (camera position in world frame)
        self.trajectory = [(0.0, 0.0, 0.0)]  # Store estimated (x, y, z) positions
        
        # Motion filtering
        self.velocity = np.zeros(3)  # Estimated velocity for motion filtering
        self.acceleration = np.zeros(3)  # Estimated acceleration for motion filtering
        self.prev_timestamps = deque(maxlen=10)  # For dt calculation
        self.prev_positions = deque(maxlen=10)  # For velocity and acceleration calculations
        self.last_update_time = time.time()
        
        # Scale estimation parameters
        self.altitude = 0.0  # Current drone altitude (from Tello)
        self.prev_altitude = 0.0  # Previous altitude
        self.scale_factor = 1.0  # Dynamic scale factor
        self.scale_window = deque(maxlen=5)  # Window for scale averaging
        
        # Keyframe selection parameters
        self.keyframe_interval = 5  # Check for new keyframe every N frames
        self.min_keyframe_distance = 0.5  # Minimum distance for new keyframe
        self.keyframes = []  # Store keyframes for loop closure
        
        # Performance parameters
        self.frame_count = 0
        self.min_features_for_motion = 15  # Increased minimum features
        self.min_tracking_ratio = 0.5  # Minimum ratio of tracked features
        self.last_keyframe_id = 0
        self.keyframe_matches_threshold = 30  # Min matches for keyframe
        
        # Track stats
        self.tracking_quality = 1.0  # Value between 0-1 indicating tracking quality
        self.drift_estimate = 0.0  # Estimated drift (increases with distance)
        
        print(f"Enhanced Visual Odometry initialized with {self.feature_detector_type} detector.")

    def update_altitude(self, altitude):
        """
        Update drone altitude from Tello for scale estimation.
        
        Args:
            altitude: Current altitude in meters from Tello
        """
        self.prev_altitude = self.altitude
        self.altitude = altitude
    
    def _preprocess_frame(self, frame):
        """
        Converts frame to grayscale and undistorts with enhanced preprocessing.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (grayscale, undistorted color frame)
        """
        try:
            # Ensure frame is contiguous
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            
            # Apply contrast enhancement (CLAHE)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            frame_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Apply undistortion
            frame_undistorted = cv2.undistort(frame_enhanced, self.K, self.D, None, self.K)
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
            
            return gray, frame_undistorted
            
        except Exception as e:
            print(f"Warning: Error in frame preprocessing: {e}. Using original frame.")
            # Fallback to basic processing
            frame_undistorted = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return gray, frame_undistorted
    
    def _estimate_scale(self, relative_t):
        """
        Estimate the scale factor for translation using multiple methods.
        
        Args:
            relative_t: Estimated relative translation vector
            
        Returns:
            float: Estimated scale factor
        """
        # Method 1: Use Tello altitude if available
        altitude_scale = 1.0
        if abs(self.altitude - self.prev_altitude) > 0.05:  # 5cm threshold
            altitude_scale = abs(self.altitude - self.prev_altitude)
        
        # Method 2: Use velocity consistency
        velocity_scale = 1.0
        if len(self.prev_positions) >= 2:
            dt = time.time() - self.last_update_time
            if dt > 0:
                # Expected translation based on current velocity
                expected_translation = np.linalg.norm(self.velocity) * dt
                # Actual translation from VO (unit vector)
                actual_translation_unit = np.linalg.norm(relative_t)
                
                if actual_translation_unit > 0:
                    velocity_scale = expected_translation / actual_translation_unit
                    # Limit scale changes to prevent jumps
                    velocity_scale = np.clip(velocity_scale, 0.5, 2.0)
        
        # Method 3: Moving average of recent scales
        combined_scale = (altitude_scale + velocity_scale) / 2.0
        self.scale_window.append(combined_scale)
        
        # Use median filter to remove outliers
        scale = np.median(self.scale_window) if len(self.scale_window) > 0 else 1.0
        
        # Apply adaptive scaling based on tracking quality
        scale = scale * self.tracking_quality
        
        # Constrain scale to reasonable bounds
        scale = np.clip(scale, 0.1, 5.0)
        
        return scale
    
    def _should_create_keyframe(self, current_frame_gray):
        """
        Determine if a new keyframe should be created based on various criteria.
        
        Returns:
            bool: True if a new keyframe should be created
        """
        # Check frame interval
        if self.frame_count % self.keyframe_interval != 0:
            return False
            
        # Check if we moved enough from last keyframe
        if self.keyframes and len(self.trajectory) > 1:
            last_keyframe_pos = self.keyframes[-1][0]  # (position, frame, kp, desc)
            current_pos = self.trajectory[-1]
            
            # Calculate distance
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(current_pos, last_keyframe_pos)))
            
            # If we haven't moved enough, don't create a new keyframe
            if distance < self.min_keyframe_distance:
                return False
                
        # Check tracking quality
        if self.tracking_quality < 0.6:  # If tracking quality is poor
            return True
            
        return True  # Create keyframe if all checks pass
    
    def _add_keyframe(self, frame_gray, position):
        """
        Add a new keyframe to the keyframe database.
        
        Args:
            frame_gray: Grayscale frame to use as keyframe
            position: Current estimated position
        """
        kp, desc = self.detector.detectAndCompute(frame_gray, None)
        
        if desc is not None and len(kp) > self.min_features_for_motion:
            # Store keyframe with position, frame, keypoints and descriptors
            self.keyframes.append((position, frame_gray.copy(), kp, desc))
            self.last_keyframe_id = len(self.keyframes) - 1
            
            # Update reference frame for tracking
            self.reference_keyframe = frame_gray.copy()
            self.reference_kp = kp
            self.reference_desc = desc
            
            print(f"Keyframe {len(self.keyframes)} created at position {position}")
            
            # Limit number of keyframes to prevent memory issues
            if len(self.keyframes) > 50:
                # Keep first keyframe and most recent ones
                self.keyframes = [self.keyframes[0]] + self.keyframes[-49:]
    
    def _check_loop_closure(self, current_kp, current_desc):
        """
        Check for loop closure opportunities among stored keyframes.
        
        Args:
            current_kp: Current keypoints
            current_desc: Current descriptors
            
        Returns:
            tuple: (found_closure, correction_R, correction_t)
        """
        # Skip if we don't have enough keyframes
        if len(self.keyframes) < 5:
            return False, None, None
            
        # Skip recent keyframes (to find actual loops)
        candidates = self.keyframes[:-4]
        
        best_matches = 0
        best_keyframe_idx = -1
        
        # Check each keyframe for potential matches
        for idx, (pos, frame, kp, desc) in enumerate(candidates):
            # Skip if current keyframe is too close to candidate
            if np.linalg.norm(np.array(self.trajectory[-1]) - np.array(pos)) < 2.0:
                continue
                
            # Match features
            try:
                matches = self.matcher.knnMatch(desc, current_desc, k=2)
                
                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) > best_matches and len(good_matches) > self.keyframe_matches_threshold:
                    best_matches = len(good_matches)
                    best_keyframe_idx = idx
            except:
                continue
        
        # If we found a good match, try to estimate transformation
        if best_keyframe_idx >= 0:
            _, frame, kp, desc = candidates[best_keyframe_idx]

            try: # Add try-except around the second match and pose recovery
                matches = self.matcher.knnMatch(desc, current_desc, k=2)

                good_matches = []
                pts_kf = []
                pts_curr = []

                # Check length before unpacking again
                for match_pair in matches:
                     if len(match_pair) == 2: # Check if we got 2 neighbors
                         m, n = match_pair # Unpack safely
                         if m.distance < 0.75 * n.distance:
                             good_matches.append(m)
                             # Ensure indices are valid
                             if m.queryIdx < len(kp) and m.trainIdx < len(current_kp):
                                 pts_kf.append(kp[m.queryIdx].pt)
                                 pts_curr.append(current_kp[m.trainIdx].pt)
                             else:
                                  print(f"Warning: Invalid match index found in loop closure pose estimation.")


                if len(good_matches) > self.keyframe_matches_threshold:
                    # ... (convert points to float32) ...
                    pts_kf = np.float32(pts_kf)
                    pts_curr = np.float32(pts_curr)

                    # Find essential matrix
                    # ... (findEssentialMat call) ...
                    E, mask = cv2.findEssentialMat(pts_curr, pts_kf, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)


                    if E is not None:
                        # Recover pose
                        # ... (recoverPose call) ...
                        _, R, t, mask_recover = cv2.recoverPose(E, pts_curr, pts_kf, self.K, mask=mask)

                        # Check if transformation is valid
                        inlier_count_loop = np.sum(mask_recover) if mask_recover is not None else 0
                        if inlier_count_loop > 20: # At least 20 inliers
                            print(f"Loop closure detected with keyframe {best_keyframe_idx} ({inlier_count_loop} inliers)")
                            return True, R, t
            except Exception as e_pose_loop:
                 print(f"Warning: Exception during loop closure pose estimation (keyframe {best_keyframe_idx}): {e_pose_loop}")
                 # --- Add traceback for detailed debugging ---
                 import traceback
                 traceback.print_exc()
                 # --- End traceback ---

        # --- END FIX 2 (Part 2) ---

        return False, None, None
    
    def _track_motion(self, pts_prev, pts_curr):
        """
        Track motion between two sets of points and handle outliers.
        
        Args:
            pts_prev: Previous points
            pts_curr: Current points
            
        Returns:
            tuple: (success, R, t, inlier_count)
        """
        # Ensure minimum number of points
        if len(pts_prev) < self.min_features_for_motion or len(pts_curr) < self.min_features_for_motion:
            return False, None, None, 0
            
        # Setup RANSAC parameters
        E, mask = cv2.findEssentialMat(
            pts_curr, pts_prev, self.K, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0  # Tighter threshold for better accuracy
        )
        
        if E is None or E.shape != (3, 3):
            return False, None, None, 0
            
        # Recover pose
        _, R, t, mask_recover = cv2.recoverPose(E, pts_curr, pts_prev, self.K, mask=mask)
        
        # Count inliers
        inlier_count = np.sum(mask_recover) if mask_recover is not None else 0
        
        # Check if motion estimation is valid
        if inlier_count < self.min_features_for_motion:
            return False, None, None, 0
            
        # Update tracking quality based on inlier ratio
        self.tracking_quality = inlier_count / len(pts_prev) if len(pts_prev) > 0 else 0
        
        return True, R, t, inlier_count
    
    def _update_motion_model(self, position):
        """
        Update velocity and acceleration estimates based on new position.
        
        Args:
            position: Current position (x, y, z)
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        if dt > 0 and len(self.prev_positions) > 0:
            # Calculate velocity
            prev_pos = np.array(self.prev_positions[-1])
            curr_pos = np.array(position)
            
            # Update velocity estimate (with smoothing)
            new_velocity = (curr_pos - prev_pos) / dt
            self.velocity = 0.7 * self.velocity + 0.3 * new_velocity
            
            # Update acceleration if we have enough positions
            if len(self.prev_positions) >= 2:
                prev_velocity = (prev_pos - np.array(self.prev_positions[-2])) / dt
                new_accel = (new_velocity - prev_velocity) / dt
                self.acceleration = 0.75 * self.acceleration + 0.2 * new_accel
        
        # Store current position and time
        self.prev_positions.append(position)
        self.prev_timestamps.append(current_time)
        self.last_update_time = current_time
    
    def _filter_motion(self, position):
        """
        Apply motion filtering to estimated position.
        
        Args:
            position: Raw estimated position
            
        Returns:
            tuple: Filtered position
        """
        if len(self.prev_positions) < 2:
            return position
            
        # Simple EKF-inspired filter
        dt = time.time() - self.last_update_time
        
        # Predict position based on velocity model
        predicted_pos = np.array(self.prev_positions[-1]) + self.velocity * dt
        
        # Weight between measurement and prediction based on tracking quality
        weight = self.tracking_quality  # Higher quality = more weight to measurement
        
        # Combine measurement with prediction
        filtered_pos = weight * np.array(position) + (1 - weight) * predicted_pos
        
        # Update drift estimate
        self.drift_estimate += 0.01 * np.linalg.norm(np.array(position) - predicted_pos)
        
        return tuple(filtered_pos)

    def process_frame(self, current_frame_color, altitude=None):
        """
        Processes a new frame to estimate motion and update trajectory.

        Args:
            current_frame_color: The new video frame (BGR format).
            altitude: Current altitude in meters from Tello (optional)

        Returns:
            tuple: (current_R, current_t, processed_frame, tracking_quality)
        """
        self.frame_count += 1
        
        # Update altitude if provided
        if altitude is not None:
            self.update_altitude(altitude)
        
        # Preprocess: Undistort and convert to grayscale
        current_frame_gray, current_frame_undistorted_color = self._preprocess_frame(current_frame_color)

        # Detect features in current frame
        kp_curr, des_curr = self.detector.detectAndCompute(current_frame_gray, None)

        # Handle the case of insufficient features
        if des_curr is None or len(kp_curr) < self.min_features_for_motion:
            print(f"Frame {self.frame_count}: Not enough features detected ({len(kp_curr) if kp_curr else 0}).")
            # Keep previous state for next frame
            self.prev_frame_gray = current_frame_gray
            self.prev_keypoints = kp_curr
            self.prev_descriptors = des_curr
            return self.current_R, self.current_t, current_frame_undistorted_color, self.tracking_quality

        estimated_motion = False
        relative_R = np.identity(3)
        relative_t = np.zeros((3, 1))

        # Check for loop closure if we have keyframes
        loop_found, loop_R, loop_t = self._check_loop_closure(kp_curr, des_curr)
        
        if loop_found:
            # Apply loop closure correction
            self.current_R = loop_R @ self.current_R
            self.current_t = self.current_t + self.current_R @ loop_t
            
            # Reset drift estimate
            self.drift_estimate = 0.0
            print("Loop closure applied - position corrected")

        # Match features with previous frame
        if self.prev_descriptors is not None:
            try:
                matches = self.matcher.knnMatch(self.prev_descriptors, des_curr, k=2)

                good_matches = []
                pts_prev = []
                pts_curr = []

                # Check length before unpacking
                for match_pair in matches:
                    if len(match_pair) == 2: # Check if we got 2 neighbors
                        m, n = match_pair # Unpack safely
                        if m.distance < 0.8 * n.distance: # Apply ratio test
                            good_matches.append(m)
                            # Ensure indices are valid before accessing points
                            if m.queryIdx < len(self.prev_keypoints) and m.trainIdx < len(kp_curr):
                                pts_prev.append(self.prev_keypoints[m.queryIdx].pt)
                                pts_curr.append(kp_curr[m.trainIdx].pt)
                            else:
                                print(f"Warning: Invalid match index found in process_frame.")
                
                if len(good_matches) >= self.min_features_for_motion:
                    pts_prev = np.float32(pts_prev)
                    pts_curr = np.float32(pts_curr)
                    
                    # Track motion
                    success, R_rel, t_rel, inlier_count = self._track_motion(pts_prev, pts_curr)
                    
                    if success:
                        estimated_motion = True
                        relative_R = R_rel
                        relative_t = t_rel
                        
                        # Estimate scale using various methods
                        absolute_scale = self._estimate_scale(relative_t)
                        
                        # Make sure translation vector has right shape
                        if relative_t.shape == (1, 3):
                            relative_t = relative_t.T
                        if relative_t.shape != (3, 1):
                            estimated_motion = False
                    else:
                        # Motion estimation failed, tracking quality decreased
                        self.tracking_quality *= 0.8
            except Exception as e:
                print(f"Frame {self.frame_count}: Error in motion estimation: {e}")
                # --- Add traceback for detailed debugging ---
                import traceback
                traceback.print_exc()
                # --- End traceback ---
                self.tracking_quality *= 0.8

        # Update trajectory if motion was estimated
        if estimated_motion:
            # Get scale (placeholder for actual scale estimation)
            absolute_scale = self._estimate_scale(relative_t)
            
            # Accumulate pose
            scaled_relative_t_world = self.current_R @ (absolute_scale * relative_t)
            self.current_t = self.current_t + scaled_relative_t_world
            self.current_R = relative_R @ self.current_R
            
            # Extract position from self.current_t
            raw_position = tuple(self.current_t.flatten())
            
            # Apply motion filtering
            filtered_position = self._filter_motion(raw_position)
            
            # Update trajectory with filtered position
            self.trajectory.append(filtered_position)
            
            # Update motion model
            self._update_motion_model(filtered_position)
            
            # Check if we should create a new keyframe
            if self._should_create_keyframe(current_frame_gray):
                self._add_keyframe(current_frame_gray.copy(), filtered_position)
        else:
            # If motion wasn't estimated, append last known position
            if self.trajectory:
                self.trajectory.append(self.trajectory[-1])
                self.tracking_quality *= 0.9  # Decrease tracking quality

        # Update previous state for next iteration
        self.prev_frame_gray = current_frame_gray
        self.prev_keypoints = kp_curr
        self.prev_descriptors = des_curr

        # Visualize tracking results on frame
        result_frame = self._visualize_tracking(current_frame_undistorted_color, kp_curr, 
                                                self.tracking_quality, self.trajectory)

        return self.current_R, self.current_t, result_frame, self.tracking_quality
    
    def _visualize_tracking(self, frame, keypoints, quality, trajectory):
        """
        Visualize tracking results on the frame.
        
        Args:
            frame: Frame to draw on
            keypoints: Current keypoints
            quality: Tracking quality (0-1)
            trajectory: Estimated trajectory
            
        Returns:
            Frame with visualization
        """
        # Draw keypoints
        result = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), 
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Draw tracking quality indicator
        quality_color = (0, int(255 * quality), int(255 * (1 - quality)))
        cv2.putText(result, f"Tracking: {quality:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)
        
        # Draw position
        if trajectory:
            x, y, z = trajectory[-1]
            cv2.putText(result, f"Pos: ({x:.2f}, {y:.2f}, {z:.2f})", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw drift estimate
        drift_color = (0, 255, 255) if self.drift_estimate < 1.0 else (0, 165, 255)
        if self.drift_estimate > 2.0:
            drift_color = (0, 0, 255)
        cv2.putText(result, f"Drift: {self.drift_estimate:.2f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, drift_color, 2)
        
        # Draw mini-map of trajectory (top-down view)
        if len(trajectory) > 1:
            map_size = 150
            map_offset = (frame.shape[1] - map_size - 10, 10)
            
            # Create minimap background
            cv2.rectangle(result, map_offset, 
                         (map_offset[0] + map_size, map_offset[1] + map_size), 
                         (0, 0, 0), -1)
            cv2.rectangle(result, map_offset, 
                         (map_offset[0] + map_size, map_offset[1] + map_size), 
                         (255, 255, 255), 1)
            
            # Extract x, z coordinates for top-down view
            points = np.array([(t[0], t[2]) for t in trajectory[-100:]])
            
            # Normalize to fit in the minimap
            if len(points) > 1:
                min_vals = np.min(points, axis=0)
                max_vals = np.max(points, axis=0)
                
                # Avoid division by zero
                range_vals = np.maximum(max_vals - min_vals, 0.1)
                
                # Scale to fit in map with padding
                padding = 0.1
                scale = (map_size * (1 - 2 * padding)) / np.max(range_vals)
                
                # Draw trajectory line
                for i in range(1, len(points)):
                    pt1 = (
                        int(map_offset[0] + padding * map_size + (points[i-1][0] - min_vals[0]) * scale),
                        int(map_offset[1] + padding * map_size + (points[i-1][1] - min_vals[1]) * scale)
                    )
                    pt2 = (
                        int(map_offset[0] + padding * map_size + (points[i][0] - min_vals[0]) * scale),
                        int(map_offset[1] + padding * map_size + (points[i][1] - min_vals[1]) * scale)
                    )
                    cv2.line(result, pt1, pt2, (0, 255, 0), 1)
                
                # Draw current position
                current_pt = (
                    int(map_offset[0] + padding * map_size + (points[-1][0] - min_vals[0]) * scale),
                    int(map_offset[1] + padding * map_size + (points[-1][1] - min_vals[1]) * scale)
                )
                cv2.circle(result, current_pt, 3, (0, 0, 255), -1)
        
        return result

    def get_trajectory(self):
        """Returns the estimated trajectory points."""
        return self.trajectory

    def get_pose(self):
        """Returns the current pose (R, t)."""
        return self.current_R, self.current_t
    
    def get_velocity(self):
        """Returns the current velocity estimate."""
        return self.velocity
    
    def get_tracking_quality(self):
        """Returns the current tracking quality (0-1)."""
        return self.tracking_quality

    def reset(self):
        """Resets the VO state completely."""
        self.prev_frame_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.reference_keyframe = None
        self.reference_kp = None
        self.reference_desc = None
        
        self.current_R = np.identity(3)
        self.current_t = np.zeros((3, 1))
        self.trajectory = [(0.0, 0.0, 0.0)]
        
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.prev_timestamps = deque(maxlen=10)
        self.prev_positions = deque(maxlen=10)
        self.last_update_time = time.time()
        
        self.altitude = 0.0
        self.prev_altitude = 0.0
        self.scale_factor = 1.0
        self.scale_window = deque(maxlen=5)
        
        self.keyframes = []
        self.last_keyframe_id = 0
        
        self.frame_count = 0
        self.tracking_quality = 1.0
        self.drift_estimate = 0.0
        
        print("Visual Odometry Reset.")