import os
import sys
import subprocess
import json
import time

# =====================================================
# NASTAVENIE OPENPOSE ROOT
# =====================================================
OPENPOSE_ROOT = r"C:\openpose"

# Možné lokácie OpenPoseDemo.exe
POSSIBLE_EXE_PATHS = [
    os.path.join(OPENPOSE_ROOT, "bin", "OpenPoseDemo.exe"),
    os.path.join(OPENPOSE_ROOT, "build", "x64", "Release", "OpenPoseDemo.exe"),
    os.path.join(OPENPOSE_ROOT, "build", "bin", "OpenPoseDemo.exe"),
]

OPENPOSE_EXE = None
for path in POSSIBLE_EXE_PATHS:
    if os.path.exists(path):
        OPENPOSE_EXE = path
        break

if OPENPOSE_EXE is None:
    print("❌ OpenPoseDemo.exe sa nenašiel.")
    print("Skontroluj, kde máš OpenPose nainštalovaný.")
    print("Skúšané cesty:")
    for p in POSSIBLE_EXE_PATHS:
        print(" -", p)
    sys.exit(1)

print("✅ OpenPoseDemo.exe nájdený:")
print(OPENPOSE_EXE)

# =====================================================
# OUTPUT DIR
# =====================================================
OUTPUT_DIR = os.path.join(OPENPOSE_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# VÝBER KAMERY
# =====================================================
print("\n0 - kamera notebooku")
print("1 - externá kamera")
CAMERA_INDEX = input("Vyber kameru: ")

# =====================================================
# SPUSTENIE OPENPOSE (EXE)
# =====================================================
cmd = [
    OPENPOSE_EXE,
    "--camera", CAMERA_INDEX,
    "--model_pose", "BODY_25",
    "--write_json", OUTPUT_DIR,
    "--display", "1",
    "--render_pose", "1"
]

print("\n▶ Spúšťam OpenPose...")
print("CMD:", " ".join(cmd))

process = subprocess.Popen(cmd)

print("\n✅ OpenPose beží")
print("CTRL + C pre ukončenie\n")

# =====================================================
# ČÍTANIE JSON (ukážka – pravé zápästie)
# =====================================================
try:
    while True:
        files = sorted(
            [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")],
            reverse=True
        )

        if files:
            latest = os.path.join(OUTPUT_DIR, files[0])
            with open(latest, "r") as f:
                data = json.load(f)

            if data.get("people"):
                pose = data["people"][0]["pose_keypoints_2d"]
                x = pose[4 * 3]
                y = pose[4 * 3 + 1]
                conf = pose[4 * 3 + 2]

                print(f"Right wrist → X:{x:.1f} Y:{y:.1f} conf:{conf:.2f}")

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\n⛔ Ukončujem OpenPose...")
    process.terminate()
