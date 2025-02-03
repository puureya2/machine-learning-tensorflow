import os
import shutil

from sklearn.model_selection import train_test_split
from src.utils import root_dir as root


input_dir = f'{root}/Data/standardized'
output_dir = f'{root}/Data/split'
categories = os.listdir(input_dir)
print(f"Found categories: {categories}")

for category in categories:
    print(f"\nProcessing category: {category}")

    class_dir = os.path.join(input_dir, category)
    images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
    print(f"Found {len(images)} images in category '{category}'.")

    train, test = train_test_split(images, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    print(f"Split sizes for '{category}': Train={len(train)}, Validate={len(val)}, Test={len(test)}")

    for split, data in zip(["train", "validate", "test"], [train, val, test]):
        split_dir = os.path.join(output_dir, split, category)
        os.makedirs(split_dir, exist_ok=True)
        print(f"Copying {len(data)} images to {split_dir}...")
        for img_path in data:
            shutil.copy(img_path, split_dir)

print("\nData splitting and copying complete!")
