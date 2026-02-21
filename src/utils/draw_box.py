"""
draw_box.py — Draw a bounding box on an image around a given (x, y) coordinate.

Usage:
    python src/draw_box.py --image path/to/image.png --x 117 --y 87
    python src/draw_box.py --image path/to/image.png --x 117 --y 87 --size 40 --color red
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw


def draw_box(
    image_path: str,
    x: int,
    y: int,
    size: int = 30,
    color: str = "red",
    thickness: int = 3,
    output_path: str = None,
) -> str:
    """Draw a square box centered on (x, y) and save the result.

    Args:
        image_path: Path to the source image.
        x: X-coordinate of the target point.
        y: Y-coordinate of the target point.
        size: Half-width/height of the box in pixels (box = 2*size × 2*size).
        color: Box outline color (name or hex, e.g. "red", "#FF0000").
        thickness: Outline thickness in pixels.
        output_path: Where to save. Defaults to <original>_boxed.<ext>.

    Returns:
        Path to the saved output image.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Bounding box corners centred on (x, y)
    left   = x - size
    top    = y - size
    right  = x + size
    bottom = y + size

    # Draw rectangle
    draw.rectangle([left, top, right, bottom], outline=color, width=thickness)

    # Draw crosshair at the exact coordinate
    draw.line([x - size, y, x + size, y], fill=color, width=1)
    draw.line([x, y - size, x, y + size], fill=color, width=1)

    # Determine output path
    if output_path is None:
        p = Path(image_path)
        output_path = str(p.with_stem(p.stem + "_boxed"))

    img.save(output_path)
    print(f"Saved: {output_path}  (box centred at x={x}, y={y}, size=±{size}px)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw a bounding box on an image")
    parser.add_argument("--image",  required=True, help="Path to input image")
    parser.add_argument("--x",      type=int, required=True, help="X coordinate")
    parser.add_argument("--y",      type=int, required=True, help="Y coordinate")
    parser.add_argument("--size",   type=int, default=30,
                        help="Half-size of the box in pixels (default: 30)")
    parser.add_argument("--color",  default="red",
                        help="Box color name or hex (default: red)")
    parser.add_argument("--thickness", type=int, default=3,
                        help="Outline thickness in pixels (default: 3)")
    parser.add_argument("--output", default=None,
                        help="Output path (default: <input>_boxed.<ext>)")
    args = parser.parse_args()

    draw_box(
        image_path=args.image,
        x=args.x,
        y=args.y,
        size=args.size,
        color=args.color,
        thickness=args.thickness,
        output_path=args.output,
    )
