"""
Predict a handwritten digit from a photo or image file.

Usage (from the project root or from src/):

    python src/predict.py path/to/photo.jpg
    python src/predict.py path/to/photo.jpg --show        # also open a preview window
    python src/predict.py photo.png --model src/model.npz

The script loads the weights and biases saved from the trained network
(``model.npz``), converts the photo to the same format as the MNIST
training images, and runs it through the network.
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageFilter, ImageOps

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.npz")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_model(path):
    """Load ``(weights, biases)`` saved with
    ``np.savez(path, *net.weights, *net.biases, sizes=net.sizes)``."""
    if not os.path.exists(path):
        sys.exit(
            f"Model file not found: {path}\n"
            "Train the network in the notebook, then save it with:\n"
            '    np.savez("model.npz", *net.weights, *net.biases, sizes=net.sizes)'
        )
    d = np.load(path)
    n = len(d["sizes"]) - 1  # number of weight matrices
    weights = [d[f"arr_{i}"] for i in range(n)]
    biases = [d[f"arr_{i}"] for i in range(n, 2 * n)]
    return weights, biases


def feedforward(weights, biases, a):
    """Run a (784, 1) column vector through the network."""
    for w, b in zip(weights, biases):
        a = sigmoid(np.dot(w, a) + b)
    return a


# --------------------------------------------------------------------------- #
# Image preprocessing (photo -> MNIST-style 28x28)
# --------------------------------------------------------------------------- #
def preprocess(path):
    """Turn a photo of a single handwritten digit into a (784, 1) vector.

    MNIST digits are white on black, scaled to fit a 20x20 box, and centred by
    centre of mass inside a 28x28 image. We copy those steps so that a photo
    looks like the training data.

    Returns ``(vector, image28)`` where ``image28`` is the 28x28 array in [0, 1]
    (useful for previews).
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # respect phone-camera rotation
    img = img.convert("L")  # grayscale

    # Work on a moderately sized copy (faster, and handwriting stays clear)
    img.thumbnail((400, 400), Image.LANCZOS)

    # 1. Remove uneven lighting and paper shadows.
    #    Estimate the background with a large max/min filter (strokes are thinner
    #    than the filter), smooth it, then keep only what differs from it (the ink).
    size = max(3, (max(img.size) // 15) | 1)  # odd filter size, ~1/15 of the image
    if np.median(np.asarray(img)) > 127:  # dark ink on light paper
        bg = img.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.GaussianBlur(size))
        ink = np.asarray(bg, dtype=np.float64) - np.asarray(img, dtype=np.float64)
    else:  # light ink on a dark background (like MNIST itself)
        bg = img.filter(ImageFilter.MinFilter(size)).filter(ImageFilter.GaussianBlur(size))
        ink = np.asarray(img, dtype=np.float64) - np.asarray(bg, dtype=np.float64)
    ink = np.clip(ink, 0, None)

    # 2. Stretch contrast to 0-255 and drop faint noise / texture
    peak = np.percentile(ink, 99.5)
    if peak < 20:
        sys.exit("Could not find a digit in the image (not enough contrast).")
    arr = np.clip(ink / peak, 0, 1) * 255
    arr[arr < 0.25 * 255] = 0

    ys, xs = np.nonzero(arr)
    if len(xs) == 0:
        sys.exit("Could not find a digit in the image (it looks blank).")

    # 3. Crop tightly around the digit
    arr = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    crop = Image.fromarray(arr.astype(np.uint8))

    # 4. Scale the longer side to 20 px, keeping the aspect ratio
    w, h = crop.size
    scale = 20.0 / max(w, h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    crop = crop.resize(new_size, Image.LANCZOS)

    # 5. Paste into the middle of a 28x28 black canvas
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(crop, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))
    out = np.asarray(canvas, dtype=np.float64) / 255.0

    # 6. Shift so the centre of mass sits at the centre (like MNIST)
    total = out.sum()
    if total > 0:
        rows, cols = np.indices(out.shape)
        cy = (rows * out).sum() / total
        cx = (cols * out).sum() / total
        shift_y = int(np.clip(round(13.5 - cy), -4, 4))
        shift_x = int(np.clip(round(13.5 - cx), -4, 4))
        out = np.roll(out, (shift_y, shift_x), axis=(0, 1))

    return out.reshape(784, 1), out


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Predict a handwritten digit from a photo.")
    parser.add_argument("image", help="path to the photo (png, jpg, ...)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="path to model.npz")
    parser.add_argument("--show", action="store_true", help="show a preview window")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"Image not found: {args.image}")

    weights, biases = load_model(args.model)
    x, image28 = preprocess(args.image)
    output = feedforward(weights, biases, x).ravel()
    prediction = int(np.argmax(output))

    print(f"\nPredicted digit: {prediction}\n")
    print("Activations:")
    for digit, value in enumerate(output):
        bar = "#" * int(round(value * 30))
        marker = "  <--" if digit == prediction else ""
        print(f"  {digit}: {value:6.3f} {bar}{marker}")

    if args.show:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        axes[0].imshow(Image.open(args.image).convert("L"), cmap="gray")
        axes[0].set_title("Your photo")
        axes[1].imshow(image28, cmap="gray")
        axes[1].set_title(f"What the network sees\nPredicted: {prediction}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
