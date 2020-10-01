import re
from dataclasses import dataclass
from typing import List


@dataclass
class Bbox:
    label: str
    coord: List[int]


@dataclass
class VocResult:
    filename: str
    width: int
    height: int
    boxes: List[Bbox]


class VocParser:
    # Filename
    _fname_pattern = r"Image filename : \"(.+)\""
    _re_fname = re.compile(_fname_pattern)

    # Image size
    _size_pattern = r"Image size \(X x Y x C\) : (\d+) x (\d+) x \d+"
    _re_size = re.compile(_size_pattern)

    # Bouding boxes
    _bbox_pattern = r"Bounding box for object \d+ \"(\w+)\" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((\d+), (\d+)\) - \((\d+), (\d+)\)"
    _re_bbox = re.compile(_bbox_pattern)

    @classmethod
    def get_filename(cls, string: str):
        return cls._re_fname.search(string).group(1)

    @classmethod
    def get_image_size(cls, string: str):
        match = cls._re_size.search(string)
        width, height = match.group(1), match.group(2)
        return (width, height)

    @classmethod
    def get_bouding_boxes(cls, string: str):
        matches = cls._re_bbox.findall(string)

        def proc_coord(coords: List):
            return [int(x) for x in coords]

        boxes = [Bbox(m[0], proc_coord(m[1:])) for m in matches]
        return boxes

    @classmethod
    def parse(cls, string: str):
        filename = cls.get_filename(string)
        width, height = cls.get_image_size(string)
        boxes = cls.get_bouding_boxes(string)
        return VocResult(filename, width, height, boxes)


# ann_sample = r"""
# # Compatible with PASCAL Annotation Version 1.00
# Image filename : "PennFudanPed/PNGImages/FudanPed00059.png"
# Image size (X x Y x C) : 576 x 369 x 3
# Database : "The Penn-Fudan-Pedestrian Database"
# Objects with ground truth : 30 { "PASpersonWalking" "PASpersonWalking" "PASpersonWalking" }
# # Note there may be some objects not included in the ground truth list for they are severe-occluded
# # or have very small size.
# # Top left pixel co-ordinates : (1, 1)
# # Details for pedestrian 1 ("PASpersonWalking")
# Original label for object 1 "PASpersonWalking" : "PennFudanPed"
# Bounding box for object 1 "PASpersonWalking" (Xmin, Ymin) - (Xmax, Ymax) : (35, 54) - (135, 330)
# Pixel mask for object 1 "PASpersonWalking" : "PennFudanPed/PedMasks/FudanPed00059_mask.png"

# # Details for pedestrian 2 ("PASpersonWalking")
# Original label for object 2 "PASpersonWalking" : "PennFudanPed"
# Bounding box for object 2 "PASpersonWalking" (Xmin, Ymin) - (Xmax, Ymax) : (159, 36) - (252, 329)
# Pixel mask for object 2 "PASpersonWalking" : "PennFudanPed/PedMasks/FudanPed00059_mask.png"

# # Details for pedestrian 3 ("PASpersonWalking")
# Original label for object 3 "PASpersonWalking" : "PennFudanPed"
# Bounding box for object 3 "PASpersonWalking" (Xmin, Ymin) - (Xmax, Ymax) : (270, 45) - (380, 328)
# Pixel mask for object 3 "PASpersonWalking" : "PennFudanPed/PedMasks/FudanPed00059_mask.png"
# """

# result = VocParser.parse(ann_sample)
# print(f"Filename: {result.filename}")
# print(f"Width x Height: {result.width} x {result.height}")
# print(f"Bounding Boxes: {result.boxes}")
# Output
# Filename: PennFudanPed/PNGImages/FudanPed00059.png
# Width x Height: 576 x 369
# Bounding Boxes: [Bbox(label='PASpersonWalking', coord=('35', '54', '135', '330')), Bbox(label='PASpersonWalking', coord=('159', '36', '252', '329')), Bbox(label='PASpersonWalking', coord=('270', '45', '380', '328'))]
