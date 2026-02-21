import sys
import os

# Add src to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from infrastructure.adapter.MSSScreenCaptureAdapter import MssScreenCaptureAdapter
    import mss
    print("mss library is installed.")
except ImportError as e:
    print(f"Error: {e}")
    print("Attempting to install mss...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mss"])
    from infrastructure.adapter.MSSScreenCaptureAdapter import MssScreenCaptureAdapter

def test_screenshot():
    adapter = MssScreenCaptureAdapter(output_dir="outputs/test_screenshots")
    print("Capturing screenshot...")
    try:
        path = adapter.capture_screenshot()
        print(f"Screenshot saved to: {path}")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print("Verification successful: Screenshot file exists and is not empty.")
        else:
            print("Verification failed: Screenshot file does not exist or is empty.")
    except Exception as e:
        print(f"An error occurred during capture: {e}")

if __name__ == "__main__":
    test_screenshot()
