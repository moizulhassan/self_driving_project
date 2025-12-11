# lane_detector.py
import cv2
import numpy as np

class LaneDetector:
    def __init__(self, src_points=None, dst_points=None):
        # default src/dst for 1280x720-ish frames; you should calibrate per camera
        if src_points is None:
            self.src = np.float32([[580,460],[700,460],[1100,720],[200,720]])
        else:
            self.src = np.array(src_points, dtype=np.float32)
        if dst_points is None:
            self.dst = np.float32([[300,0],[980,0],[980,720],[300,720]])
        else:
            self.dst = np.array(dst_points, dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def process(self, frame):
        # returns overlay image and lane coordinates in original image
        h,w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        edges = cv2.Canny(blur, 50, 150)
        # mask roi
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (int(0.1*w), h),
            (int(0.45*w), int(0.6*h)),
            (int(0.55*w), int(0.6*h)),
            (int(0.9*w), h)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        cropped = cv2.bitwise_and(edges, mask)
        # warp
        warped = cv2.warpPerspective(cropped, self.M, (w,h), flags=cv2.INTER_LINEAR)
        # sliding window / histogram
        histogram = np.sum(warped[ h//2: , : ], axis=0)
        midpoint = int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # sliding windows
        nwindows = 9
        window_height = int(h//nwindows)
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []
        for window in range(nwindows):
            win_y_low = h - (window+1)*window_height
            win_y_high = h - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        if len(leftx)>0 and len(lefty)>0:
            left_fit = np.polyfit(lefty, leftx, 2)
        else:
            left_fit = None
        if len(rightx)>0 and len(righty)>0:
            right_fit = np.polyfit(righty, rightx, 2)
        else:
            right_fit = None
        # create an image to draw lanes on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        ploty = np.linspace(0, h-1, h)
        if left_fit is not None:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        else:
            left_fitx = None
        if right_fit is not None:
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        else:
            right_fitx = None
        if left_fitx is not None and right_fitx is not None:
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (w,h))
        result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
        return result, (left_fit, right_fit)
