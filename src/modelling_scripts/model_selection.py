from keras.src.applications.densenet import DenseNet121
from keras.src.applications.mobilenet_v3 import MobileNetV3Large
from keras.src.applications.resnet import ResNet50


# Load pre-trained models without their top layers
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
densenet121 = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
mobilenetv3 = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

print("Pre-trained models loaded successfully!")
