"""Entry point to generate ground truth."""

from training import generate_tracks
from training import generate_contours

INPUT_PATH = "C:/Users/Nedsh/Documents/CS/Project/DatasetRaw/ucsdpeds/vidf_jpg"
OUTPUT_PATH = "C:/Users/Nedsh/Documents/CS/Project/DatasetLabelled/position_contours"
IN_DIR = "C:/Users/Nedsh/Documents/CS/Project/DatasetLabelled/position_contours"
OUT_DIR = "C:/Users/Nedsh/Documents/CS/Project/DatasetLabelled/position_contours_tracks"

generate_contours.track_all_in_folder(INPUT_PATH, OUTPUT_PATH)
generate_tracks.track_all_in_folder(IN_DIR, OUT_DIR)
