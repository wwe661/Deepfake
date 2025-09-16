import cv2


def preprocess_image(file, target_size=(224, 224)):
    import cv2
    import numpy as np

    # Read file bytes
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR

    debug_images = {}

    # Original
    debug_images["original"] = img.copy()

    # --- Step 1: CLAHE ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    debug_images["clahe"] = img_clahe.copy()

    # --- Step 2: Sharpening ---
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    img_sharpened = cv2.filter2D(img_clahe, -1, sharpen_kernel)
    debug_images["sharpened"] = img_sharpened.copy()

    # --- Step 3: Resize ---
    img_resized = cv2.resize(img_sharpened, target_size)
    debug_images["resized"] = img_resized.copy()

    # --- Step 4: Convert BGR → RGB & Normalize ---
    img_final = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_final = img_final.astype("float32") / 255.0
    processed = np.expand_dims(img_final, axis=0)  # for model

    return processed, debug_images

img,imgs = preprocess_image(open("fake.png", "rb"))

for k,v in imgs.items():
    from matplotlib import pyplot as plt

    plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
    plt.title(k)
    plt.show()
    
plt.imshow(img)