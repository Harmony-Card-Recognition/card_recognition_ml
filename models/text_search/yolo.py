import os, sys

PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)

from PIL import Image
from typing import Tuple

# different functions to identify different parts of an image of a card.
# the output is in this format (left_x_val, top_y_val, right_x_val, bottom_y_val)
# use a fast yolo model to identify these parts of the card image
def get_title_rectangle(image:Image) -> Tuple[int, int, int, int]:
    pass
def get_type_line_rectangle(image:Image) -> Tuple[int, int, int, int]:
    pass
def get_oracle_text_rectangle(image:Image) -> Tuple[int, int, int, int]:
    pass
def get_artist_rectangle(image:Image) -> Tuple[int, int, int, int]:
    pass
def get_power_toughness_rectangle(image:Image) -> Tuple[int, int, int, int]:
    pass
def get_mana_cost_rectangle(image:Image) -> Tuple[int, int, int, int]:
    pass