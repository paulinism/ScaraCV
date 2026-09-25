"""
take_photo.py
─────────────
Captures a photo from DroidCam (or any webcam) and saves it
as photo.jpg in the same folder where this script is located.

Usage:
  1. Open DroidCam on your phone and the DroidCam client on your PC.
  2. Run this script from VS Code or a terminal:
         python take_photo.py
  3. A live preview window will open:
       SPACE  →  capture and save photo.jpg
       ESC    →  cancel

If you have multiple cameras, change CAMERA_INDEX:
  0 = built-in laptop webcam (usually)
  1 = first external camera / DroidCam USB
  2 = second external camera / DroidCam WiFi
  (run find_cameras.py if you don't know which one to use)

Dependencies:
  pip install opencv-python
"""

import cv2
import os

# ── CONFIGURATION ──────────────────────────────────────────────────────────
CAMERA_INDEX = 1
OUTPUT_PATH = "photo.jpg"
WINDOW_TITLE = "DroidCam  —  SPACE: capture   ESC: cancel"
# ──────────────────────────────────────────────────────────────────────────


def find_available_cameras(max_index=5):
    """Displays which camera indices are available."""
    print("Searching for available cameras...")
    found = []

    for i in range(max_index):
        cap = cv2.VideoCapture(i)

        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  ✓ Index {i}  —  {w}×{h}")
            found.append(i)
            cap.release()
        else:
            print(f"  ✗ Index {i}  —  unavailable")

    return found


def take_photo(camera_index=CAMERA_INDEX, output_path=OUTPUT_PATH):
    """
    Opens the selected camera, shows a live preview,
    and saves a captured image.

    Returns:
        str: path of the saved image
        None: if capture was canceled or failed
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"\n[ERROR] Could not open camera at index {camera_index}.")
        print("  → Run find_available_cameras() first to check available indices.")
        print("  → Make sure DroidCam is running before executing this script.")
        return None

    # Suggested resolution for DroidCam (adjust if needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nCamera opened (index {camera_index})  —  {w}×{h}")
    print("Press [SPACE] to capture or [ESC] to cancel.\n")

    captured_path = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Could not read frame. Check the DroidCam connection.")
            break

        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE → save image
            script_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(script_dir, output_path)

            cv2.imwrite(full_path, frame)
            captured_path = full_path

            print(f"  ✓ Photo saved to: {full_path}")
            break

        elif key == 27:  # ESC → cancel
            print("  Capture canceled.")
            break

    cap.release()
    cv2.destroyAllWindows()

    return captured_path


# ── MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Uncomment the next line if you want to see available cameras:
    # find_available_cameras()

    result = take_photo()

    if result:
        print(f"\n  Image ready for the pipeline: {result}")
        print("  Now run figure_detection.py")
    else:
        print("\n  No image was saved.")