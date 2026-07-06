import os
import shutil
from tkinter import Tk, filedialog

from meme_assets import GESTURE_FILES, MEMES_DIR

IMAGE_FILE_TYPES = [
    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
    ("JPEG files", "*.jpg *.jpeg"),
    ("PNG files", "*.png"),
    ("All files", "*.*"),
]


def choose_mode():
    print("Choose how you want to upload cat meme photos:")
    print("1. Use one photo for all gestures")
    print("2. Choose a different photo for each gesture")

    while True:
        choice = input("Enter 1 or 2: ").strip() or "1"
        if choice in {"1", "2"}:
            return choice
        print("Please enter 1 or 2.")


def copy_image(source_path, target_name):
    os.makedirs(MEMES_DIR, exist_ok=True)
    target_path = os.path.join(MEMES_DIR, target_name)
    shutil.copy2(source_path, target_path)
    print(f"Saved: {target_path}")


def select_image(dialog_root, title):
    return filedialog.askopenfilename(
        title=title,
        filetypes=IMAGE_FILE_TYPES,
        parent=dialog_root,
    )


def upload_one_image_for_all(dialog_root):
    source_path = select_image(dialog_root, "Select a cat meme photo")
    if not source_path:
        print("No file selected. Nothing was uploaded.")
        return

    for target_name in GESTURE_FILES.values():
        copy_image(source_path, target_name)

    print("One cat meme photo was copied to all gesture slots.")


def upload_different_images(dialog_root):
    uploaded_count = 0
    for gesture_name, target_name in GESTURE_FILES.items():
        source_path = select_image(
            dialog_root,
            f"Select a cat meme photo for {gesture_name}",
        )
        if not source_path:
            print(f"Skipped: {gesture_name}")
            continue

        copy_image(source_path, target_name)
        uploaded_count += 1

    if uploaded_count == 0:
        print("No files were selected. Nothing was uploaded.")
    else:
        print(f"Uploaded {uploaded_count} meme image(s).")


def main():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    choice = choose_mode()
    if choice == "1":
        upload_one_image_for_all(root)
    else:
        upload_different_images(root)

    root.destroy()


if __name__ == "__main__":
    main()
