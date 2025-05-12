# -*- coding: utf-8 -*-
"""underwater object detection.ipynb

"""

//pip install opencv-python numpy

import cv2
import numpy as np
from IPython.display import display
from google.colab import files
import io
from PIL import Image

# Upload file
uploaded = files.upload()

for fname in uploaded.keys():
    # Read image
    image = Image.open(io.BytesIO(uploaded[fname]))
    image = np.array(image.convert('RGB'))  # Ensure it's RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize (optional)
    image = cv2.resize(image, (640, 480))

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Detect water (blue-green shades)
    lower_water = np.array([80, 40, 40])
    upper_water = np.array([130, 255, 255])
    water_mask = cv2.inRange(hsv, lower_water, upper_water)

    # Invert to get objects
    object_mask = cv2.bitwise_not(water_mask)

    # Make yellow object highlight
    yellow = np.zeros_like(image)
    yellow[:] = (0, 255, 255)
    highlighted = cv2.bitwise_and(yellow, yellow, mask=object_mask)

    # Create final image with black background
    black_background = np.zeros_like(image)
    final_result = cv2.add(black_background, highlighted)

    # Convert BGR to RGB for display
    display_image = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)

    # Show output
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(display_image)
    plt.title("Objects in Yellow, Water is Black")
    plt.axis("off")

    plt.show()

import cv2
import numpy as np
from IPython.display import display
from google.colab import files
import io
from PIL import Image

# Upload file
uploaded = files.upload()

for fname in uploaded.keys():
    # Read image
    image = Image.open(io.BytesIO(uploaded[fname]))
    image = np.array(image.convert('RGB'))  # Ensure it's RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize (optional)
    image = cv2.resize(image, (640, 480))

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Detect water (blue-green shades)
    lower_water = np.array([80, 40, 40])
    upper_water = np.array([130, 255, 255])
    water_mask = cv2.inRange(hsv, lower_water, upper_water)

    # Invert to get objects
    object_mask = cv2.bitwise_not(water_mask)

    # Make yellow object highlight
    yellow = np.zeros_like(image)
    yellow[:] = (0, 255, 255)
    highlighted = cv2.bitwise_and(yellow, yellow, mask=object_mask)

    # Create final image with black background
    black_background = np.zeros_like(image)
    final_result = cv2.add(black_background, highlighted)

    # Convert BGR to RGB for display
    display_image = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)

    # Show output
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(display_image)
    plt.title("Objects in Yellow, Water is Black")
    plt.axis("off")

    plt.show()

import cv2
import numpy as np
from IPython.display import display
from google.colab import files
import io
from PIL import Image

# Upload file
uploaded = files.upload()

for fname in uploaded.keys():
    # Read image
    image = Image.open(io.BytesIO(uploaded[fname]))
    image = np.array(image.convert('RGB'))  # Ensure it's RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize (optional)
    image = cv2.resize(image, (640, 480))

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Detect water (blue-green shades)
    lower_water = np.array([80, 40, 40])
    upper_water = np.array([130, 255, 255])
    water_mask = cv2.inRange(hsv, lower_water, upper_water)

    # Invert to get objects
    object_mask = cv2.bitwise_not(water_mask)

    # Make yellow object highlight
    yellow = np.zeros_like(image)
    yellow[:] = (0, 255, 255)
    highlighted = cv2.bitwise_and(yellow, yellow, mask=object_mask)

    # Create final image with black background
    black_background = np.zeros_like(image)
    final_result = cv2.add(black_background, highlighted)

    # Convert BGR to RGB for display
    display_image = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)

    # Show output
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(display_image)
    plt.title("Objects in Yellow, Water is Black")
    plt.axis("off")

    plt.show()

import cv2
import numpy as np
from IPython.display import display
from google.colab import files
import io
from PIL import Image

# Upload file
uploaded = files.upload()

for fname in uploaded.keys():
    # Read image
    image = Image.open(io.BytesIO(uploaded[fname]))
    image = np.array(image.convert('RGB'))  # Ensure it's RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize (optional)
    image = cv2.resize(image, (640, 480))

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Detect water (blue-green shades)
    lower_water = np.array([80, 40, 40])
    upper_water = np.array([130, 255, 255])
    water_mask = cv2.inRange(hsv, lower_water, upper_water)

    # Invert to get objects
    object_mask = cv2.bitwise_not(water_mask)

    # Make yellow object highlight
    yellow = np.zeros_like(image)
    yellow[:] = (0, 255, 255)
    highlighted = cv2.bitwise_and(yellow, yellow, mask=object_mask)

    # Create final image with black background
    black_background = np.zeros_like(image)
    final_result = cv2.add(black_background, highlighted)

    # Convert BGR to RGB for display
    display_image = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)

    # Show output
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(display_image)
    plt.title("Objects in Yellow, Water is Black")
    plt.axis("off")

    plt.show()
