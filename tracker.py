# tracker.py
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import time

def iou(bb_test, bb_gt):
    # bb = [x1,y1,x2,y2]
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    area2 = (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1])
    union = area1+area2-inter
    if union==0:
        return 0
    return inter/union

class Track:
    def __init__(self, bbox, track_id):
        # bbox: x1,y1,x2,y2
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        dt = 1.
        # state: x, y, s, r, vx, vy, vs  (we'll approximate)
        self.kf.F = np.eye(7)
        for i in range(4):
            self.kf.F[i, i+3 if i<3 else i] = 1
        self.kf.H = np.zeros((4,7))
        self.kf.H[0,0]=1
        self.kf.H[1,1]=1
        self.kf.H[2,2]=1
        self.kf.H[3,3]=1
        x1,y1,x2,y2 = bbox
        cx = (x1+x2)/2.
        cy = (y1+y2)/2.
        s = (x2-x1)*(y2-y1)
        r = (x2-x1)/(y2-y1+1e-6)
        self.kf.x[:4,0] = np.array([cx, cy, s, r])
        self.kf.P *= 10.
        self.time_since_update = 0
        self.id = track_id
        self.history = []
        self.hits = 1
        self.age = 0
        self.bbox = bbox

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update>0:
            self.hits = 0
        self.time_since_update += 1
        cx, cy, s, r = self.kf.x[:4,0]
        w = np.sqrt(s*r)
        h = s / (w+1e-6)
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        self.bbox = [int(x1),int(y1),int(x2),int(y2)]
        return self.bbox

    def update(self, bbox):
        x1,y1,x2,y2 = bbox
        cx = (x1+x2)/2.
        cy = (y1+y2)/2.
        s = (x2-x1)*(y2-y1)
        r = (x2-x1)/(y2-y1+1e-6)
        z = np.array([cx, cy, s, r])
        self.kf.update(z)
        self.time_since_update = 0
        self.hits += 1
        self.history.append(self.bbox)

class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        # detections: list of [x1,y1,x2,y2,score,class]
        trks = [t.predict() for t in self.tracks]
        matched, unmatched_dets, unmatched_trks = self.associate(detections, trks)
        # update matched
        for det_idx, trk_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx][:4])
        # create new tracks for unmatched_dets
        for i in unmatched_dets:
            t = Track(detections[i][:4], self.next_id)
            self.next_id += 1
            self.tracks.append(t)
        # remove dead tracks
        removed = []
        for t in self.tracks:
            if t.time_since_update > self.max_age:
                removed.append(t)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        # prepare output: tracks with id and bbox
        out = []
        for t in self.tracks:
            if t.hits >= self.min_hits or t.age < self.min_hits:
                out.append((t.id, t.bbox))
        return out

    def associate(self, detections, trks):
        if len(trks)==0:
            return [], list(range(len(detections))), []
        iou_matrix = np.zeros((len(detections), len(trks)), dtype=np.float32)
        for d,det in enumerate(detections):
            for t,trk in enumerate(trks):
                iou_matrix[d,t] = iou(det[:4], trk)
        # Hungarian on -iou
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched, unmatched_dets, unmatched_trks = [], [], []
        for d in range(len(detections)):
            if d not in row_ind: unmatched_dets.append(d)
        for t in range(len(trks)):
            if t not in col_ind: unmatched_trks.append(t)
        # filter by iou threshold
        for r,c in zip(row_ind, col_ind):
            if iou_matrix[r,c] < self.iou_threshold:
                unmatched_dets.append(r)
                unmatched_trks.append(c)
            else:
                matched.append((r,c))
        return matched, unmatched_dets, unmatched_trks

