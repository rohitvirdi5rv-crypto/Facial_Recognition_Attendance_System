import subprocess
import sys
import os

try:
    app_path = os.path.join(os.path.dirname(sys.executable), "app.py")
    
    subprocess.run([
        "streamlit",
        "run",
        app_path
    ], check=True)

except Exception as e:
    print("ERROR:", e)
    input("Press Enter to exit...")