from bing_image_downloader import downloader
from src.utils import root_dir as root


# Define the classes and number of images per class
classes = {
    "corn crop field": 500,  # Number of images to download for corn
    "wheat crop field": 500,  # Number of images to download for wheat
    "soybean crop field": 500  # Number of images to download for soybean
}

# Output directory for the dataset
output_dir = f'{root}/Data/raw'

# Download images for each class
n = 1
for crop, limit in classes.items():

    downloader.download(
        crop,
        limit=limit,
        output_dir=output_dir,
        adult_filter_off=True,  # Keep adult filter on
        force_replace=False,  # Do not overwrite if the folder already exists
        timeout=120,  # Timeout for each connection
        verbose=True  # Show progress
    )

    print(f"{n}: Downloaded '{crop}'...")
    n = n + 1

