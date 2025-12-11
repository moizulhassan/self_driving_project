# speed_estimator.py
import numpy as np
import cv2
import time

class SpeedEstimator:
    def __init__(self, pts_src=None, pts_dst=None, ppm=None):
        """
        ppm: pixels per meter scale on bird-eye view. If None, user must calibrate.
        pts_src/dst: for homography if needed
        """
        self.ppm = ppm  # pixels per meter
        self.prev_positions = {}  # track_id -> (x_meter, y_meter, timestamp)
        self.min_time_delta = 0.1

    def set_ppm(self, ppm):
        self.ppm = ppm

    def pixel_to_meter(self, pixel_xy):
        # convert pixel coords (x,y) in bird-eye to meters using ppm
        if self.ppm is None:
            raise ValueError("Set ppm (pixels-per-meter) via calibration.")
        x_m = pixel_xy[0] / self.ppm
        y_m = pixel_xy[1] / self.ppm
        return x_m, y_m

    def estimate_speed(self, track_id, bbox, timestamp, birdseye_transform=None):
        # bbox center
        x1,y1,x2,y2 = bbox
        cx = (x1+x2)/2.0
        cy = (y1+y2)/2.0
        # optionally warp to birdseye
        if birdseye_transform is not None:
            p = np.array([[[cx,cy]]], dtype='float32')
            p_trans = cv2.perspectiveTransform(p, birdseye_transform)[0][0]
            cx,cy = p_trans[0], p_trans[1]
        if self.ppm is None:
            return None
        x_m, y_m = self.pixel_to_meter((cx,cy))
        if track_id in self.prev_positions:
            prev_x, prev_y, prev_t = self.prev_positions[track_id]
            dt = timestamp - prev_t
            if dt < self.min_time_delta:
                return None
            dist = np.sqrt((x_m - prev_x)**2 + (y_m - prev_y)**2)
            speed_m_s = dist / dt
            speed_kmh = speed_m_s * 3.6
            self.prev_positions[track_id] = (x_m, y_m, timestamp)
            return speed_kmh
        else:
            self.prev_positions[track_id] = (x_m, y_m, timestamp)
            return None
