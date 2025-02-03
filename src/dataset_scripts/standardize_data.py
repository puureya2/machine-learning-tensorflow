import os
from PIL import Image
from src.utils import root_dir


input_dir = f'{root_dir}/Data/raw'
output_dir = f'{root_dir}/Data/standardized'
size = (224, 224)

x = 1
y = 1

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(root, file)
            img = Image.open(img_path)

            # Convert to RGB mode if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img = img.resize(size)
            class_name = os.path.basename(root)
            save_dir = os.path.join(output_dir, class_name)
            os.makedirs(save_dir, exist_ok=True)

            # Ensure file extension is valid for saving
            filename, ext = os.path.splitext(file)
            new_file = f"{filename}.jpg"  # Force saving as .jpg
            img.save(os.path.join(save_dir, new_file))

            print(f"{x}. {new_file} standardized")
            x += 1

    print(f"{y}. {files} standardized")
    y += 1
