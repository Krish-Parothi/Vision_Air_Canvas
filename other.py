"""
Virtual Air Pen (Selfie Camera Mode)
File: virtual_air_pen_selfie.py
Author: ChatGPT (GPT-5)

Description:
- Uses MediaPipe Hands + OpenCV to track hand in real time.
- Selfie camera view (mirrored).
- Two-finger gesture toggles drawing ON/OFF.
- Optional GPU acceleration for coordinate smoothing via PyTorch.

Controls:
- ✌️ Two fingers up → Toggle drawing mode ON/OFF
- 🧹 Press 'c' → Clear canvas
- 💾 Press 's' → Save current drawing
- ❌ Press 'q' or 'ESC' → Quit
"""

import argparse
import time
import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    raise ImportError("mediapipe is required. Install via `pip install mediapipe`")

USE_TORCH = False
try:
    import torch
    USE_TORCH = True
except Exception:
    USE_TORCH = False


def millis():
    return int(round(time.time() * 1000()))


class SmoothFilter:
    def __init__(self, alpha=0.6, device=None):
        self.alpha = alpha
        self.prev = None
        self.device = device

    def reset(self):
        self.prev = None

    def apply(self, pt):
        if pt is None:
            self.prev = None
            return None
        if self.prev is None:
            self.prev = np.array(pt, dtype=np.float32)
            return tuple(self.prev.tolist())

        if USE_TORCH and self.device is not None:
            t_prev = torch.tensor(self.prev, device=self.device)
            t_pt = torch.tensor(pt, device=self.device)
            t_out = self.alpha * t_pt + (1 - self.alpha) * t_prev
            out = t_out.cpu().numpy()
            self.prev = out
            return (float(out[0]), float(out[1]))
        else:
            out = self.alpha * np.array(pt, dtype=np.float32) + (1 - self.alpha) * self.prev
            self.prev = out
            return (float(out[0]), float(out[1]))


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def fingers_up(hand_landmarks):
    """Return list [index, middle, ring, pinky] True/False for each finger up"""
    lm = hand_landmarks.landmark
    fingers = []
    # Index finger
    fingers.append(lm[8].y < lm[6].y)
    # Middle
    fingers.append(lm[12].y < lm[10].y)
    # Ring
    fingers.append(lm[16].y < lm[14].y)
    # Pinky
    fingers.append(lm[20].y < lm[18].y)
    return fingers


def main(args):
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    have_cuda = False
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            have_cuda = True
    except Exception:
        have_cuda = False

    use_gpu_ops = args.use_gpu and USE_TORCH and torch.cuda.is_available()
    device = torch.device('cuda') if use_gpu_ops else None

    print(f"GPU ops: {use_gpu_ops}, OpenCV CUDA available: {have_cuda}")

    ret, frame = cap.read()
    if not ret:
        print("Camera not available")
        return

    h, w = frame.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    filter = SmoothFilter(alpha=0.6, device=device)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # ✅ Only one hand used
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    drawing = False
    prev_point = None
    color = (0, 0, 255)
    thickness = 6
    fps_time = time.time()
    fps = 0

    last_seen_time = time.time()
    hand_lost_timeout = 2.5
    last_two_finger_state = False  # For toggle control

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ✅ Flip the frame horizontally for selfie view
            frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            pen_point = None

            # Hand detected
            if results.multi_hand_landmarks:
                last_seen_time = time.time()
                hand_landmarks = results.multi_hand_landmarks[0]

                # detect 2 fingers up
                fingers = fingers_up(hand_landmarks)
                two_fingers_up = fingers[0] and fingers[1] and not fingers[2] and not fingers[3]

                # Toggle drawing ON/OFF only when 2 fingers are raised again
                if two_fingers_up and not last_two_finger_state:
                    drawing = not drawing
                    print(f"Drawing mode toggled to: {drawing}")

                last_two_finger_state = two_fingers_up

                # index tip = pen nib
                lm = hand_landmarks.landmark
                ix = int(lm[8].x * w)
                iy = int(lm[8].y * h)
                pen_point = (ix, iy)

            else:
                # hand temporarily lost
                if time.time() - last_seen_time > hand_lost_timeout:
                    pen_point = None

                last_two_finger_state = False

            # Smooth pen point
            smooth_pt = None
            if pen_point is not None and drawing:
                smooth_pt = filter.apply(pen_point)
            else:
                if pen_point is None:
                    filter.reset()
                smooth_pt = None

            # Draw on canvas
            if smooth_pt is not None:
                if prev_point is None:
                    prev_point = smooth_pt
                cv2.line(canvas, (int(prev_point[0]), int(prev_point[1])),
                         (int(smooth_pt[0]), int(smooth_pt[1])),
                         color, thickness, cv2.LINE_AA)
                prev_point = smooth_pt
            else:
                prev_point = None

            # Overlay canvas
            overlay = cv2.addWeighted(frame, 0.6, canvas, 0.4, 0)
            status_text = f"Drawing Mode: {drawing} | FPS: {fps:.1f}"
            cv2.putText(overlay, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if drawing:
                cv2.putText(overlay, "Drawing: ON (Two-Finger Toggle)", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(overlay, "Drawing: OFF (Two-Finger Toggle)", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("🖊️ Virtual Air Pen (Selfie Mode)", overlay)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (now - fps_time)) if fps_time and now != fps_time else 0
            fps_time = now

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:  # ESC or Q to quit
                break
            elif key == ord('c'):
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
            elif key == ord('s'):
                cv2.imwrite(f"airpen_{millis()}.png",
                            cv2.addWeighted(frame, 0.6, canvas, 0.4, 0))

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Virtual Air Pen (Selfie Camera)')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for smoothing ops (requires PyTorch + CUDA)')
    args = parser.parse_args()
    main(args)
