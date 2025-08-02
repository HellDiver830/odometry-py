"""
trajectory_only_fix.py
CPU-одометрия с «гейтом скорости» — синия линия, без «стрелы в бесконечность».

▪ Режем translation, если:
   1) средний параллакс фич < 2 px  → считаем кадр стоячим;
   2) мгновенная скорость > 2×скользящего среднего за 30 кадров → выброс.

pip install opencv-contrib-python==4.8.1.78 numpy mss matplotlib
"""
import cv2, numpy as np, matplotlib.pyplot as plt
from mss import mss
from collections import deque

#Ызахват окна 
REG = {"top": 0, "left": 0, "width": 640, "height": 360}
FX = FY = 450.0
CX, CY = REG["width"]/2, REG["height"]/2
K = np.array([[FX,0,CX],[0,FY,CY],[0,0,1]], dtype=np.float64)

orb = cv2.ORB_create(2000)
bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
sct = mss()

prev_gray = prev_kp = prev_desc = None
R_w = np.eye(3, dtype=np.float64)
t_w = np.zeros((3,1), dtype=np.float64)
poses_xyz = []
speed_hist = deque(maxlen=30)    # скользящее окно скоростей

#Matplotlib setup
plt.ion()
fig, ax = plt.subplots(figsize=(5,4))
ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_aspect("equal")

print("▶  Ctrl-C или закрой окно Matplotlib, чтобы выйти.")
try:
    while True:
        frame = np.array(sct.grab(REG))[:,:,:3]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            prev_kp, prev_desc = orb.detectAndCompute(gray, None)
            continue

        kp, desc = orb.detectAndCompute(gray, None)
        if desc is None or prev_desc is None or len(kp) < 8:
            prev_gray, prev_kp, prev_desc = gray, kp, desc
            continue

        matches = bf.match(prev_desc, desc)
        if len(matches) < 8:
            prev_gray, prev_kp, prev_desc = gray, kp, desc
            continue

        matches = sorted(matches, key=lambda m:m.distance)[:200]
        pts0 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
        pts1 = np.float32([kp[m.trainIdx].pt      for m in matches])

        E, _ = cv2.findEssentialMat(pts0, pts1, K, cv2.RANSAC, 0.999, 1.0)
        if E is None:
            prev_gray, prev_kp, prev_desc = gray, kp, desc
            continue
        _, R, t, mask = cv2.recoverPose(E, pts0, pts1, K)

        # ── фильтр 1: параллакс < 2 px  → t = 0
        inl = mask.ravel().astype(bool)
        parallax = np.mean(np.linalg.norm(pts0[inl] - pts1[inl], axis=1)) if inl.any() else 0
        if parallax < 2.0:
            t[:] = 0

        # ── фильтр 2: выброс по скорости
        inst_speed = np.linalg.norm(t)
        mean_speed = np.mean(speed_hist) if speed_hist else 0
        if mean_speed and inst_speed > 2 * mean_speed:
            t[:] = 0
        speed_hist.append(inst_speed)

        # ── обновляем глобальную позу
        R_w = R @ R_w
        t_w = t_w + (-R_w.T @ t)
        poses_xyz.append(t_w.flatten())

        # ── обновляем график каждые 5 кадров
        if len(poses_xyz) % 5 == 0:
            ax.cla(); ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_aspect("equal")
            ps = np.array(poses_xyz)
            if len(ps):
                ax.plot(ps[:,0], ps[:,2], c="blue")
            plt.pause(0.001)

        prev_gray, prev_kp, prev_desc = gray, kp, desc

except KeyboardInterrupt:
    print("\n⏹  Завершено.")
finally:
    plt.ioff(); plt.show()
