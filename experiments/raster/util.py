from typing import List, Tuple
import torch

def rectangles_to_logits(rects: List[Tuple], picture_size) -> torch.Tensor:
    logits = torch.zeros(picture_size, picture_size)
    for rect in rects:
        x1, y1, x2, y2 = rect
        logits[y1:y2+1, x1:x2+1] = 1
    return logits.view(-1)

def logits_to_bitmap(logits: torch.Tensor, picture_size) -> str:
    logits = logits.view(picture_size, picture_size)
    output = ""
    for i in range(picture_size):
        for j in range(picture_size):
            output += "{:.2f}  ".format(logits[i][j])
        output += "\n"
    return output
