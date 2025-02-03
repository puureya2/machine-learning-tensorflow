import os
from keras.src.utils import image_dataset_from_directory
from src.plot_scripts.data_visualization import ModelPlotter
from src.utils import root_dir as root


# Paths
model_path = f'{root}/Models/densenet121_model.keras'
history_path = f'{root}/Models/densenet121_training_history.pkl'
test_data_path = f'{root}/Data/split/test'
image_size = (224, 224)
batch_size = 64

# Load test dataset
test_dataset = image_dataset_from_directory(
    test_data_path,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False,
    label_mode="categorical"
)

# Initialize ModelPlotter
model_plotter = ModelPlotter(
    model_path=model_path,
    test_dataset=test_dataset,
    model_name="DenseNet121",
    history_path=history_path
)

# Generate and save plots
model_plotter.plot_confusion_matrix()
model_plotter.plot_training_metrics()
