from collections import deque
import numpy as np

class MovingAverage:
    """A helper class to compute the moving average of a value sequence."""
    def __init__(self, size=8):
        self.size = size
        self.queue = deque(maxlen=size)
        
    def add(self, val):
        self.queue.append(val)
        
    def get(self):
        if not self.queue:
            return 0.0
        return sum(self.queue) / len(self.queue)


class BaseExerciseTracker:
    """
    Modular Base class for tracking exercises using a finite state machine,
    moving averages for smoothing, and coordinate/velocity processing.
    """
    def __init__(self, window_size=8):
        self.window_size = window_size
        self.rep_count = 0
        self.state = "STANDING"       # Current FSM state
        self.direction = "STATIONARY"   # Current direction of torso movement
        self.reset()
        
    def reset(self):
        """Resets all metrics, counts, and smoothers to default."""
        self.rep_count = 0
        self.state = "STANDING"
        self.direction = "STATIONARY"
        
    def update(self, *args, **kwargs):
        """Processes a single frame. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement update().")


class SquatTracker(BaseExerciseTracker):
    """
    A robust motion analysis system and finite state machine for squat rep counting.
    Smoothes joint angles and vertical coordinates using moving averages,
    determines vertical velocity of the torso, and transitions states when
    velocity direction and joint angles agree.
    """
    def __init__(self, window_size=8):
        super().__init__(window_size)
        
    def reset(self):
        super().reset()
        # Smoothers for knee and hip angles
        self.left_knee_smoother = MovingAverage(self.window_size)
        self.right_knee_smoother = MovingAverage(self.window_size)
        self.avg_knee_smoother = MovingAverage(self.window_size)
        self.left_hip_smoother = MovingAverage(self.window_size)
        self.right_hip_smoother = MovingAverage(self.window_size)
        self.avg_hip_smoother = MovingAverage(self.window_size)
        
        # Smoothers for elbow angles
        self.left_elbow_smoother = MovingAverage(self.window_size)
        self.right_elbow_smoother = MovingAverage(self.window_size)
        
        # Smoothers for vertical positions (to compute velocity)
        self.shoulder_y_smoother = MovingAverage(self.window_size)
        self.hip_y_smoother = MovingAverage(self.window_size)
        
        # History for velocity calculations
        self.prev_shoulder_y = None
        self.prev_hip_y = None
        
        # Current smoothed metrics for UI display
        self.smoothed_left_knee = 180.0
        self.smoothed_right_knee = 180.0
        self.smoothed_avg_knee = 180.0
        self.smoothed_left_hip = 180.0
        self.smoothed_right_hip = 180.0
        self.smoothed_avg_hip = 180.0
        self.smoothed_left_elbow = 180.0
        self.smoothed_right_elbow = 180.0
        
        # State threshold angles (in degrees)
        self.angle_standing = 160.0    # Above this is standing
        self.angle_descending = 145.0  # Below this is starting squat
        self.angle_bottom = 100.0      # Below this is bottom
        self.angle_ascending = 115.0   # Above this is returning
        
        # Torso vertical velocity threshold (normalized y-coordinates per frame)
        self.velocity_threshold = 0.0015
        
    def update(self, l_knee, r_knee, l_hip, r_hip, shoulder_y, hip_y, l_elbow=None, r_elbow=None):
        """
        Updates smoothers, calculates torso velocity and direction, 
        and updates the finite state machine.
        
        Args:
            l_knee (float or None): Left knee angle
            r_knee (float or None): Right knee angle
            l_hip (float or None): Left hip angle
            r_hip (float or None): Right hip angle
            shoulder_y (float or None): Average shoulder Y coordinate (normalized)
            hip_y (float or None): Average hip Y coordinate (normalized)
            l_elbow (float or None): Left elbow angle
            r_elbow (float or None): Right elbow angle
        """
        # 1. Update Angle Smoothers
        if l_knee is not None:
            self.left_knee_smoother.add(l_knee)
        if r_knee is not None:
            self.right_knee_smoother.add(r_knee)
        if l_knee is not None and r_knee is not None:
            self.avg_knee_smoother.add((l_knee + r_knee) / 2.0)
        elif l_knee is not None:
            self.avg_knee_smoother.add(l_knee)
        elif r_knee is not None:
            self.avg_knee_smoother.add(r_knee)
            
        if l_hip is not None:
            self.left_hip_smoother.add(l_hip)
        if r_hip is not None:
            self.right_hip_smoother.add(r_hip)
        if l_hip is not None and r_hip is not None:
            self.avg_hip_smoother.add((l_hip + r_hip) / 2.0)
        elif l_hip is not None:
            self.avg_hip_smoother.add(l_hip)
        elif r_hip is not None:
            self.avg_hip_smoother.add(r_hip)
            
        if l_elbow is not None:
            self.left_elbow_smoother.add(l_elbow)
        if r_elbow is not None:
            self.right_elbow_smoother.add(r_elbow)
            
        # Retrieve current smoothed values
        self.smoothed_left_knee = self.left_knee_smoother.get() if l_knee is not None else 180.0
        self.smoothed_right_knee = self.right_knee_smoother.get() if r_knee is not None else 180.0
        self.smoothed_avg_knee = self.avg_knee_smoother.get()
        self.smoothed_left_hip = self.left_hip_smoother.get() if l_hip is not None else 180.0
        self.smoothed_right_hip = self.right_hip_smoother.get() if r_hip is not None else 180.0
        self.smoothed_avg_hip = self.avg_hip_smoother.get()
        self.smoothed_left_elbow = self.left_elbow_smoother.get() if l_elbow is not None else 180.0
        self.smoothed_right_elbow = self.right_elbow_smoother.get() if r_elbow is not None else 180.0
        
        # 2. Update Position Smoothers and Calculate Velocity
        current_shoulder_y = None
        current_hip_y = None
        
        if shoulder_y is not None:
            self.shoulder_y_smoother.add(shoulder_y)
            current_shoulder_y = self.shoulder_y_smoother.get()
            
        if hip_y is not None:
            self.hip_y_smoother.add(hip_y)
            current_hip_y = self.hip_y_smoother.get()
            
        # Determine movement direction based on vertical velocity
        direction = "STATIONARY"
        v_shoulder = 0.0
        v_hip = 0.0
        
        if self.prev_shoulder_y is not None and current_shoulder_y is not None:
            v_shoulder = current_shoulder_y - self.prev_shoulder_y
            
        if self.prev_hip_y is not None and current_hip_y is not None:
            v_hip = current_hip_y - self.prev_hip_y
            
        if current_shoulder_y is not None and current_hip_y is not None:
            # Average velocity of both shoulders and hips
            v_avg = (v_shoulder + v_hip) / 2.0
            
            # Since coordinate Y increases downwards:
            # v_avg > threshold => moving down (DESCENDING)
            # v_avg < -threshold => moving up (ASCENDING)
            if v_avg > self.velocity_threshold:
                direction = "DESCENDING"
            elif v_avg < -self.velocity_threshold:
                direction = "ASCENDING"
            else:
                direction = "STATIONARY"
                
        self.direction = direction
        
        # Save positions for next frame comparison
        if current_shoulder_y is not None:
            self.prev_shoulder_y = current_shoulder_y
        if current_hip_y is not None:
            self.prev_hip_y = current_hip_y
            
        # 3. Finite State Machine Transitions (Agreement of Velocity and Angles)
        knee_angle = self.smoothed_avg_knee
        
        if self.state == "STANDING":
            # Transition to descending if knee is bending AND velocity is downwards
            if knee_angle < self.angle_descending and self.direction == "DESCENDING":
                self.state = "DESCENDING"
                
        elif self.state == "DESCENDING":
            # Transition to bottom if knee angle is below threshold and velocity stops/slows down
            if knee_angle < self.angle_bottom:
                self.state = "BOTTOM"
            # Return to standing if they stand up before reaching bottom
            elif knee_angle > self.angle_standing and self.direction == "ASCENDING":
                self.state = "STANDING"
                
        elif self.state == "BOTTOM":
            # Transition to ascending if knee extends AND velocity is upwards
            if knee_angle > self.angle_ascending and self.direction == "ASCENDING":
                self.state = "ASCENDING"
                
        elif self.state == "ASCENDING":
            # Transition back to standing to complete a rep
            if knee_angle > self.angle_standing and (self.direction == "ASCENDING" or self.direction == "STATIONARY"):
                self.rep_count += 1
                self.state = "STANDING"
            # Bends knees again (falls back to descending) if they drop back down
            elif knee_angle < self.angle_bottom and self.direction == "DESCENDING":
                self.state = "DESCENDING"
