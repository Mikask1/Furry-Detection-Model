from PIL import Image
import os
folder_path = os.getcwd()+"/images/non-furry/"

index = 0
for path in os.listdir(folder_path):
    path = folder_path+path

    img = Image.open(path)
    img = img.resize((300, 300))
    img = img.convert("RGB")

    os.remove(path)
    path = os.path.splitext(path)[0]
    img.save(folder_path+f"{index}.jpg", "JPEG", quality=50, optimize=True)
    index += 1
