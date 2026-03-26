import argparse
import random
from tqdm import tqdm
import torch
import os
import pickle
import json

from viz import plot_bitmap

def generate_training_datapoint(picture_size, valid_rects, num_shapes):
    rects = torch.randint(0, len(valid_rects), (num_shapes,))

    bitmap = torch.zeros((picture_size, picture_size))

    for rect_index in rects:
        (x1, y1, x2, y2) = valid_rects[rect_index]
        bitmap[x1:x2+1, y1:y2+1] = 1.

    # reshape to have one initial channel that represents color
    bitmap.reshape(1, picture_size, picture_size)
    return (rects, bitmap)

def main():
    parser = argparse.ArgumentParser(description="Generate training data for rasterizer")
    # where to put output data (no default)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--picture-size", type=int, default=4)
    parser.add_argument("--num-shapes", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=1000)
    args = parser.parse_args()

    def rect_is_valid(rect):
        (x1, y1, x2, y2) = rect
        if x1 < x2 and y1 < y2:
            return True
        elif (x1 < x2 and y1 <= y2) or (x1 <= x2 and y1 < y2):
            return True
        else:
            return False

    valid_rects = [
            (x1, y1, x2, y2)
            for x1 in range(args.picture_size)
            for y1 in range(args.picture_size)
            for x2 in range(args.picture_size)
            for y2 in range(args.picture_size)
            if rect_is_valid((x1, y1, x2, y2))
            ]

    print("There are {} valid rectangles of size {}".format(len(valid_rects), args.picture_size))

    bitmaps = torch.zeros((args.num_samples, 1, args.picture_size, args.picture_size))
    rects = torch.zeros((args.num_samples, args.num_shapes), dtype=torch.int64)
    for i in tqdm(range(args.num_samples)):
        (rect_indices, resulting_bitmap) = generate_training_datapoint(args.picture_size, valid_rects, args.num_shapes)
        bitmaps[i] = resulting_bitmap
        rects[i] = rect_indices

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(bitmaps, os.path.join(args.output_dir, "bitmaps.pt"))
    torch.save(rects, os.path.join(args.output_dir, "rects.pt"))
    pickle.dump(valid_rects, open(os.path.join(args.output_dir, "valid_rects.pkl"), "wb"))

    # print bitmaps to file
    with open(os.path.join(args.output_dir, "bitmaps.txt"), "w") as f:
        for i in range(args.num_samples):
            f.write("\nBitmap {}\n".format(i))
            f.write(plot_bitmap(bitmaps[i][0]))

main()
