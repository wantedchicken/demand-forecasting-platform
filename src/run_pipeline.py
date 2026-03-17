import subprocess
import time
import sys

steps = [
    ("Model Training", [sys.executable, "src/train_save.py"]),
]

print("\n🚀 Starting Demand Forecasting Pipeline\n")

for step_name, command in steps:
    print(f"🔹 Running: {step_name}")
    start = time.time()

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"❌ Error in step: {step_name}")
        break

    end = time.time()
    print(f"✅ Completed {step_name} in {round(end-start, 2)} seconds\n")

print("🎉 Pipeline Finished!")