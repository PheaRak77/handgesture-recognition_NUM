# -*- coding: utf-8 -*-
"""
find_cameras.py  -  Fast, unbuffered camera scanning utility.
Supports USB webcam index scanning and custom DroidCam WiFi stream probing.

Usage:
    venv\\Scripts\\python.exe scripts\\find_cameras.py
"""

import sys
import cv2

# Force stdout to flush immediately
sys.stdout.reconfigure(encoding='utf-8')

def print_flush(text=""):
    print(text, flush=True)

print_flush("=" * 60)
print_flush("       🔍 KHMER GESTURE - CAMERA SCANNER & CONFIG 🔍")
print_flush("=" * 60)
print_flush("This utility will quickly scan for connected cameras (indices 0-4)\n"
            "and optionally test DroidCam WiFi connections.\n")

# 1. Scan local camera indices
print_flush("--- Step 1: Scanning Local Camera Indices (0-4) ---")
print_flush("Probing with CAP_ANY for speed...")

found_indices = []
for idx in range(5):
    print_flush(f"  Probing local index {idx}...")
    # Using CAP_ANY is the most compatible default
    cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps_label = f"{fps:.0f}fps" if fps > 0 else "N/A"
            print_flush(f"    ⭐ [FOUND] Index {idx} works! ({w}x{h} @ {fps_label})")
            found_indices.append(idx)
        else:
            print_flush(f"    ⚠️  [WARNING] Index {idx} opens but failed to read frames.")
        cap.release()
    else:
        print_flush(f"    ❌ Index {idx} is not available.")

print_flush("\n" + "=" * 60)

# 2. Ask user for DroidCam IP if they want to test WiFi connection
print_flush("\n--- Step 2: Test DroidCam WiFi Stream (Optional) ---")
print_flush("If your iPhone and laptop are on the same WiFi, DroidCam allows")
print_flush("streaming video directly via an IP address.")
print_flush("Look at the DroidCam screen on your iPhone for the 'WiFi IP' (e.g. 192.168.1.15)\n")

print_flush("Would you like to test a DroidCam WiFi connection?")
print_flush("If yes, run this script with your IP, for example:")
print_flush("  venv\\Scripts\\python.exe scripts\\find_cameras.py --ip 192.168.1.15\n")

# Check if IP argument is passed
ip_arg = None
for i, arg in enumerate(sys.argv):
    if arg == "--ip" and i + 1 < len(sys.argv):
        ip_arg = sys.argv[i + 1]

if ip_arg:
    # Remove any http:// or /video if user pasted full URL
    clean_ip = ip_arg.replace("http://", "").replace("/video", "").replace("/video.force", "")
    if ":" not in clean_ip:
        wifi_url = f"http://{clean_ip}:4747/video"
    else:
        wifi_url = f"http://{clean_ip}/video"
        
    print_flush(f"Testing DroidCam WiFi URL: {wifi_url}...")
    cap = cv2.VideoCapture(wifi_url)
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print_flush(f"    ⭐ [SUCCESS] DroidCam WiFi connected successfully! ({w}x{h})")
            print_flush(f"\n🚀 To run the Khmer Gesture app with this WiFi camera:")
            print_flush(f"    venv\\Scripts\\python.exe main.py --camera {wifi_url}")
        else:
            print_flush("    ❌ DroidCam URL opened, but failed to grab a video frame.")
        cap.release()
    else:
        print_flush("    ❌ Connection failed. Check that DroidCam is open on your iPhone")
        print_flush("       and both devices are connected to the exact same WiFi network.")
else:
    # Summary of findings and launch command
    print_flush("--- Summary of Findings ---")
    if found_indices:
        print_flush(f"Active camera indices found: {found_indices}")
        print_flush(f"  • Your Laptop Camera is likely: Index {found_indices[0]}")
        if len(found_indices) > 1:
            print_flush(f"  • Your iPhone (via DroidCam Client USB) is likely: Index {found_indices[-1]}")
            print_flush(f"\n🚀 Run the project using your iPhone camera:")
            print_flush(f"    venv\\Scripts\\python.exe main.py --camera {found_indices[-1]}")
        else:
            print_flush("\n🚀 Run the project using your Laptop camera:")
            print_flush(f"    venv\\Scripts\\python.exe main.py --camera {found_indices[0]}")
            print_flush("\n💡 Tip: To use your iPhone, connect DroidCam (USB or WiFi client) first, or test the WiFi IP.")
    else:
        print_flush("❌ No working cameras were detected.")
        print_flush("   Please connect a webcam or start your DroidCam connection.")

print_flush("\n" + "=" * 60)
