from training import generate_tracks

INPUT_PATH = "C:/Users/Nedsh/Documents/CS/Project/DatasetRaw/ucsdpeds/vidf_jpg"
OUTPUT_PATH = "C:/Users/Nedsh/Documents/CS/Project/DatasetLabelled/online_track_positions"
generate_tracks.track_all_in_folder(INPUT_PATH, OUTPUT_PATH)
