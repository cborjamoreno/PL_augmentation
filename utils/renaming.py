import os
import sys

def rename_images(directory):
    for filename in os.listdir(directory):
        src = os.path.join(directory, filename)
        if filename.endswith(".JPG"):
            dst = os.path.join(directory, filename + ".jpg")
        elif filename.endswith(".jpg"):
            dst = os.path.join(directory, filename[:-4] + ".JPG.jpg")
        else:
            dst = os.path.join(directory, os.path.splitext(filename)[0] + ".JPG.jpg")
        os.rename(src, dst)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python renaming.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    rename_images(directory)
