"""Sample tests."""
# from segmentation import batch_mask
# IN_DIR = "C:/Users/Nedsh/Documents/CS/Project/DatasetRaw/ucsdpeds/vidf_jpg/vidf1_33_000.y"
# OUT_DIR = "C:/Users/Nedsh/Documents/CS/Project/DatasetRaw/mask_ucsdpeds"
IN_DIR = "./example/raw"
# batch_mask.massImageMask(IN_DIR, OUT_DIR)

# from Segmentation import segmentation

# from karlman_filter.track_test import test
# test()

from visualisation.online import online

online(IN_DIR, 1, 8, 10, 5)
