# Make sure to install required packages:
# pip install opencv-python mediapipe numpy
import time
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import mediapipe as mp


# -----------------------------
# Small helper for rotating a 2D point around a center (de-roll)
# -----------------------------
def rotate_around(p: np.ndarray, c: np.ndarray, rad: float) -> np.ndarray:
    t = p - c
    cs, sn = np.cos(rad), np.sin(rad)
    return np.array([cs * t[0] - sn * t[1], sn * t[0] + cs * t[1]], dtype=np.float32) + c


def inter_ocular(L: np.ndarray, R: np.ndarray) -> float:
    return float(max(1.0, np.linalg.norm(L - R)))


# -----------------------------
# MediaPipe indices (Face Mesh = 468 points)
# We only need eye centers + mouth corners (equivalents of dlib 68 indices).
#
# - dlib mouth corners: 48 (left), 54 (right)
# - MediaPipe mouth corners: 61 (left), 291 (right)
#
# For eye center estimation (used for roll + IOD):
# Use two stable corners per eye and average them (similar role to averaging dlib eye ranges).
# -----------------------------
MP_LEFT_EYE_CORNER_IDS = [33, 133]
MP_RIGHT_EYE_CORNER_IDS = [362, 263]
MP_LEFT_MOUTH_CORNER_ID = 61
MP_RIGHT_MOUTH_CORNER_ID = 291


class Stage(Enum):
    STAGE_ALIGN = 0
    STAGE_HOLD_STILL = 1
    STAGE_PROMPT_SMILE = 2
    STAGE_EVALUATE = 3


@dataclass
class Config:
    # Camera
    cam_w: int = 1280
    cam_h: int = 720
    mirror_view: bool = True

    # Guide oval parameters (fractions of cam w/h)
    guide_center_y_frac: float = 0.45
    guideA_frac: float = 0.18
    guideB_frac: float = 0.28

    # Stability / calibration 
    HOLD_STILL_SECONDS: float = 1.5
    MOVE_THRESH_NORM: float = 0.012

    # Smile + asymmetry 
    smoothing_alpha: float = 0.2
    SMILE_MIN: float = 0.06
    ASYM_THRESHOLD: float = 0.35
    SMILE_HOLD_SECONDS: float = 0.6

    # Guide fraction requirement 
    inside_fraction_needed: float = 0.92


class StrokeSmileApp:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # MediaPipe FaceMesh (pretrained internal models)
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Guide oval sized like in setup()
        self.guide_center = np.array([cfg.cam_w * 0.5, cfg.cam_h * cfg.guide_center_y_frac], dtype=np.float32)
        self.guideA = cfg.cam_w * cfg.guideA_frac
        self.guideB = cfg.cam_h * cfg.guideB_frac

        # State machine variables (direct ports)
        self.stage = Stage.STAGE_ALIGN

        self.prev_center = np.zeros(2, dtype=np.float32)
        self.have_prev_center = False
        self.stable_time = 0.0

        self.base_left_corner = np.zeros(2, dtype=np.float32)
        self.base_right_corner = np.zeros(2, dtype=np.float32)
        self.baseIOD = 1.0
        self.calibrated = False

        self.smile_intensity = 0.0
        self.asymmetry_score = 0.0

        self.smile_hold_time = 0.0
        self.abnormal = False

        self.last_t = time.time()

    # -----------------------------
    # Detection: returns pts (468,2) in image coords, or None
    # -----------------------------
    def detect_landmarks(self, frame_bgr: np.ndarray):
        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            return None

        face = res.multi_face_landmarks[0]
        pts = np.array([(lm.x * w, lm.y * h) for lm in face.landmark], dtype=np.float32)
        return pts

    # -----------------------------
    # - compute eye "centers"
    # - compute roll from eye line
    # - deroll all points around the eye midpoint
    # -----------------------------
    def get_derolled_keypoints(self, pts: np.ndarray):
        left_eye_c = pts[MP_LEFT_EYE_CORNER_IDS].mean(axis=0)
        right_eye_c = pts[MP_RIGHT_EYE_CORNER_IDS].mean(axis=0)
        center = 0.5 * (left_eye_c + right_eye_c)

        d = right_eye_c - left_eye_c
        roll = float(np.arctan2(d[1], d[0]))

        pts_der = np.empty_like(pts)
        for i in range(pts.shape[0]):
            pts_der[i] = rotate_around(pts[i], center, -roll)

        left_eye_der = rotate_around(left_eye_c, center, -roll)
        right_eye_der = rotate_around(right_eye_c, center, -roll)

        # FaceCenter = center (non-derolled eye midpoint)
        face_center = center
        return pts_der, left_eye_der, right_eye_der, face_center

    # -----------------------------
    # checks *all* landmarks. Here we can also check all 468 points.
    # -----------------------------
    def mostly_inside_guide(self, pts: np.ndarray, fraction_needed: float = 0.92) -> bool:
        nx = (pts[:, 0] - self.guide_center[0]) / self.guideA
        ny = (pts[:, 1] - self.guide_center[1]) / self.guideB
        inside = (nx * nx + ny * ny) <= 1.0
        return inside.mean() >= fraction_needed

    # -----------------------------
    # updateStability function to track stable head position
    # -----------------------------
    def update_stability(self, center: np.ndarray, iod: float, inside_guide: bool, dt: float):
        if not inside_guide:
            self.have_prev_center = False
            self.stable_time = 0.0
            return

        if not self.have_prev_center:
            self.prev_center = center.copy()
            self.have_prev_center = True
            self.stable_time = 0.0
            return

        move_norm = float(np.linalg.norm(center - self.prev_center) / max(1.0, iod))
        self.prev_center = center.copy()

        if move_norm < self.cfg.MOVE_THRESH_NORM:
            self.stable_time += dt
        else:
            self.stable_time = 0.0

    # -----------------------------
    # updateSmileMetrics function to track smile intensity and asymmetry
    # -----------------------------
    def update_smile_metrics(self, pts_der: np.ndarray, iod: float):
        if (not self.calibrated) or (pts_der.shape[0] <= max(MP_LEFT_MOUTH_CORNER_ID, MP_RIGHT_MOUTH_CORNER_ID)):
            # decay while not ready (matches ofLerp(..., 0, 0.2))
            self.smile_intensity = (1.0 - 0.2) * self.smile_intensity
            self.asymmetry_score = (1.0 - 0.2) * self.asymmetry_score
            return

        L = pts_der[MP_LEFT_MOUTH_CORNER_ID]
        R = pts_der[MP_RIGHT_MOUTH_CORNER_ID]

        iod = max(1.0, iod)

        # baseline-current
        # this gives positive values for smiles (corners raised)
        left_raise = float((self.base_left_corner[1] - L[1]) / iod)
        right_raise = float((self.base_right_corner[1] - R[1]) / iod)

        intensity = max(0.0, 0.5 * (abs(left_raise) + abs(right_raise)))
        denom = 0.5 * (abs(left_raise) + abs(right_raise)) + 1e-6
        asym = abs(left_raise - right_raise) / denom

        a = self.cfg.smoothing_alpha
        self.smile_intensity = (1.0 - a) * self.smile_intensity + a * intensity
        self.asymmetry_score = (1.0 - a) * self.asymmetry_score + a * asym

    # -----------------------------
    # Reset all state
    # -----------------------------
    def reset_all(self):
        self.stage = Stage.STAGE_ALIGN
        self.calibrated = False
        self.abnormal = False
        self.smile_intensity = 0.0
        self.asymmetry_score = 0.0
        self.smile_hold_time = 0.0
        self.stable_time = 0.0
        self.have_prev_center = False

    # -----------------------------
    # Main per-frame update
    # -----------------------------
    def step(self, frame_bgr: np.ndarray):
        now = time.time()
        dt = now - self.last_t
        self.last_t = now

        pts = self.detect_landmarks(frame_bgr)

        # If no face, decay metrics and go back to ALIGN (unless evaluating)
        if pts is None:
            self.smile_intensity = (1.0 - 0.08) * self.smile_intensity
            self.asymmetry_score = (1.0 - 0.08) * self.asymmetry_score
            self.have_prev_center = False
            if self.stage != Stage.STAGE_EVALUATE:
                self.stage = Stage.STAGE_ALIGN
            return None, None, False

        pts_der, left_eye_c, right_eye_c, face_c = self.get_derolled_keypoints(pts)
        iod = inter_ocular(left_eye_c, right_eye_c)
        inside = self.mostly_inside_guide(pts_der, self.cfg.inside_fraction_needed)

        # State machine transitions
        if self.stage == Stage.STAGE_ALIGN:
            self.have_prev_center = False
            self.stable_time = 0.0
            self.smile_hold_time = 0.0
            self.calibrated = False
            if inside:
                self.stage = Stage.STAGE_HOLD_STILL

        elif self.stage == Stage.STAGE_HOLD_STILL:
            self.update_stability(face_c, iod, inside, dt)
            if not inside:
                self.stage = Stage.STAGE_ALIGN
            elif self.stable_time >= self.cfg.HOLD_STILL_SECONDS:
                self.base_left_corner = pts_der[MP_LEFT_MOUTH_CORNER_ID].copy()
                self.base_right_corner = pts_der[MP_RIGHT_MOUTH_CORNER_ID].copy()
                self.baseIOD = iod
                self.calibrated = True
                self.stage = Stage.STAGE_PROMPT_SMILE

        elif self.stage == Stage.STAGE_PROMPT_SMILE:
            if not inside:
                self.stage = Stage.STAGE_ALIGN
            else:
                self.update_smile_metrics(pts_der, self.baseIOD)

                if self.smile_intensity >= self.cfg.SMILE_MIN:
                    self.smile_hold_time += dt
                else:
                    self.smile_hold_time = 0.0

                if self.smile_hold_time >= self.cfg.SMILE_HOLD_SECONDS:
                    self.abnormal = (self.asymmetry_score > self.cfg.ASYM_THRESHOLD)
                    self.stage = Stage.STAGE_EVALUATE

        elif self.stage == Stage.STAGE_EVALUATE:
            pass

        return pts, pts_der, inside

    # -----------------------------
    # Drawing
    # -----------------------------
    def draw(self, frame_bgr: np.ndarray, pts: np.ndarray, inside: bool, fps: float):
        # Guide oval (green if aligned / red otherwise)
        is_ok_color = inside and (self.stage != Stage.STAGE_ALIGN)
        color = (40, 220, 120) if is_ok_color else (230, 70, 70)

        center = (int(self.guide_center[0]), int(self.guide_center[1]))
        axes = (int(self.guideA), int(self.guideB))
        cv2.ellipse(frame_bgr, center, axes, 0, 0, 360, color, 3)

        # Landmarks overlay: MediaPipe has 468 points; draw a light subset to keep it readable.
        if pts is not None:
            for idx in (
                MP_LEFT_EYE_CORNER_IDS
                + MP_RIGHT_EYE_CORNER_IDS
                + [MP_LEFT_MOUTH_CORNER_ID, MP_RIGHT_MOUTH_CORNER_ID]
            ):
                x, y = pts[idx].astype(int)
                cv2.circle(frame_bgr, (x, y), 3, (0, 255, 255), -1)

        # HUD / instructions
        x, y = 20, 30
        if self.stage == Stage.STAGE_ALIGN:
            line1 = "Align your head inside the oval."
            line2 = "Keep your head roughly upright."
            line3 = ""
        elif self.stage == Stage.STAGE_HOLD_STILL:
            prog = min(1.0, self.stable_time / self.cfg.HOLD_STILL_SECONDS) * 100.0
            line1 = "Hold still..."
            line2 = "Capturing neutral baseline."
            line3 = f"Progress: {prog:.0f}%"
        elif self.stage == Stage.STAGE_PROMPT_SMILE:
            line1 = "Show your teeth (smile)!"
            line2 = "Hold for a moment..."
            line3 = f"Smile intensity: {self.smile_intensity:.3f}  |  Asym: {self.asymmetry_score:.3f}"
        else:
            line1 = "Result:"
            line2 = "ABNORMALITY DETECTED" if self.abnormal else "NORMAL"
            line3 = "Press [R] to restart"

        def put(txt, yy, scale=0.7):
            if not txt:
                return
            cv2.putText(frame_bgr, txt, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)

        put(line1, y)
        put(line2, y + 22)
        put(line3, y + 44)

        # Big banner for final result 
        if self.stage == Stage.STAGE_EVALUATE:
            banner = "ABNORMALITY DETECTED" if self.abnormal else "NORMAL"
            bcol = (60, 60, 230) if self.abnormal else (60, 200, 120)
            cv2.putText(frame_bgr, banner, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.3, bcol, 3, cv2.LINE_AA)

        # FPS
        cv2.putText(frame_bgr, f"Framerate : {fps:.1f}", (20, frame_bgr.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    cfg = Config()
    app = StrokeSmileApp(cfg)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_h)

    fps = 0.0
    last = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if cfg.mirror_view:
            frame = cv2.flip(frame, 1)

        pts, pts_der, inside = app.step(frame)

        now = time.time()
        dt = now - last
        last = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        app.draw(frame, pts, inside, fps)
        cv2.imshow("Stroke Smile Asymmetry (MediaPipe)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break
        if key in (ord('r'), ord('R')):
            app.reset_all()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
