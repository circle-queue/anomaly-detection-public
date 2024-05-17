import io
import sys

import IPython.display
import numpy as np
import torch
from PIL import Image

IS_NOTEBOOK = hasattr(sys, "ps1")
if IS_NOTEBOOK:
    # Display arrays as images for convenience

    # source: https://stackoverflow.com/questions/26649716/how-to-show-pil-image-in-ipython-notebook
    def display_pil_image(arr):
        """Displayhook function for PIL Images, rendered as PNG."""

        try:
            b = io.BytesIO()
            arr_to_img(arr).save(b, format="png")
            ip_img = IPython.display.Image(data=b.getvalue(), format="png", embed=True)
        except Exception:
            print("Didn't pretty print image")
            return  # default print

        return ip_img._repr_png_()

    ipy = IPython.get_ipython()  # type: ignore
    png_formatter = ipy.display_formatter.formatters["image/png"]
    png_formatter.for_type(np.ndarray, display_pil_image)
    png_formatter.for_type(torch.Tensor, display_pil_image)


def arr_to_img(arr: np.ndarray | torch.Tensor) -> Image.Image:
    """
    Coerces a numpy array or torch tensor to a PIL Image.
    If the image is not of type uint8, it is scaled to [0, 255].
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().detach().numpy()

    arr = arr.squeeze()
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3:
        if 2 != np.argmin(arr.shape):
            arr = arr.transpose(1, 2, 0)  # Channel-axis is probably first, e.g. Tensors
    else:
        return  # Not an image

    if any(size < 10 for size in arr.shape[:2]):
        return

    if arr.dtype != "uint8":
        arr = 255 * (arr.astype(float) - arr.min()) / ((arr.max() - arr.min()) or 1)
        arr = arr.clip(0, 255).astype("uint8")
    print(f"{arr.shape = }")
    return Image.fromarray(arr)


if __name__ == "__main__":
    IPython.display.display(
        np.zeros((50, 50)),  # Should be black
        torch.ones((50, 50)),  # Should be black
        np.diag([i for i in range(50)]),  # Should be a diagonal gradient
        (np.ones((50, 50, 3)) * 255).astype("uint8"),  # Should be white
    )
