import os
import sys
import subprocess

# =====================================================
# ROOT – sem daj LEN priečinok, kde máš rozbalený OpenPose ZIP
# (nie bin, nie openpose/bin, ale koreň balíka)
# =====================================================
OPENPOSE_ROOT = r"C:\Users\jakub\Downloads\openpose-1.7.0-binaries-win64-cpu-python3.7-flir-3d"

# =====================================================
# AUTOMATICKÉ HĽADANIE OpenPoseDemo.exe
# =====================================================
def find_openpose_exe(root):
    for root_dir, dirs, files in os.walk(root):
        if "OpenPoseDemo.exe" in files:
            return os.path.join(root_dir, "OpenPoseDemo.exe")
    return None

OPENPOSE_EXE = find_openpose_exe(OPENPOSE_ROOT)

if OPENPOSE_EXE is None:
    print("❌ OpenPoseDemo.exe sa nenašiel.")
    print("Skontroloval som celý priečinok:")
    print(OPENPOSE_ROOT)
    sys.exit(1)

print("✅ OpenPoseDemo.exe nájdený:")
print(OPENPOSE_EXE)

# =====================================================
# VÝBER KAMERY
# =====================================================
print("\n0 - kamera notebooku")
print("1 - externá kamera")
CAMERA_INDEX = input("Vyber kameru: ")

# =====================================================
# OPENPOSE PRÍKAZ – BODY + FACE (ako na obrázku)
# =====================================================
cmd = [
    OPENPOSE_EXE,
    "--camera", CAMERA_INDEX,
    "--model_pose", "BODY_25",
    "--face",
    "--display", "1",
    "--render_pose", "1"
]

print("\n▶ Spúšťam OpenPose...")
print("CMD:", " ".join(cmd))

# =====================================================
# SPUSTENIE
# =====================================================
subprocess.Popen(cmd)
