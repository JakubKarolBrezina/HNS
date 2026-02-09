import os
import sys
import subprocess
import time

# =====================================================
# OPENPOSE ROOT
# =====================================================
OPENPOSE_ROOT = r"C:\Users\jakub\Downloads\openpose-1.7.0-binaries-win64-cpu-python3.7-flir-3d\openpose"

OPENPOSE_EXE = os.path.join(OPENPOSE_ROOT, "bin", "OpenPoseDemo.exe")
MODEL_FOLDER = os.path.join(OPENPOSE_ROOT, "models")

# =====================================================
# KONTROLY
# =====================================================
if not os.path.exists(OPENPOSE_EXE):
    print("❌ OpenPoseDemo.exe sa nenašiel")
    sys.exit(1)

if not os.path.exists(MODEL_FOLDER):
    print("❌ models folder sa nenašiel")
    sys.exit(1)

print("✅ OpenPoseDemo.exe:", OPENPOSE_EXE)
print("✅ models:", MODEL_FOLDER)

# =====================================================
# VÝBER KAMERY
# =====================================================
print("\n0 - kamera notebooku")
print("1 - externá kamera")
CAMERA_INDEX = input("kamera: ")

# =====================================================
# OPENPOSE PRÍKAZ
# =====================================================
cmd = [
    OPENPOSE_EXE,
    "--camera", CAMERA_INDEX,
    "--camera_resolution", "640x480",
    "--model_pose", "BODY_25",
    "--model_folder", MODEL_FOLDER,
    "--face",
    "--display", "1",
    "--render_pose", "1"
]

print("\n▶ Spúšťam OpenPose...")
print("CMD:", " ".join(cmd))
print("\n🟢 OpenPose by sa mal otvoriť v novom okne.")

# =====================================================
# SPUSTENIE + ČAKANIE
# =====================================================
process = subprocess.Popen(
    cmd,
    cwd=OPENPOSE_ROOT,
    creationflags=subprocess.CREATE_NEW_CONSOLE
)

print("\n⏳ Python teraz čaká. Zatvor OpenPose okno alebo stlač CTRL+C.\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n⛔ Ukončujem OpenPose...")
    process.terminate()
