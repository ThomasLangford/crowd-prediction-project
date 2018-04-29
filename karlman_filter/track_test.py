from segmentation.segmentation import MaskInterface
from .tracker_old import Tracker
from scipy.ndimage.measurements import center_of_mass
import cv2
import os
import numpy as np

# IN_DIR = "C:/Users/Nedsh/Documents/CS/Project/DatasetRaw/ucsdpeds/vidf_jpg/vidf4_33_014.y"
# IN_DIR = "C:/Users/Nedsh/Documents/CS/Project/DatasetRaw/ucsdpeds/vidf_jpg/vidf6_33_018.y"
IN_DIR = "C:/Users/Nedsh/Documents/CS/Project/DatasetRaw/ucsdpeds/vidf_jpg/vidf1_33_000.y"

def get_jpg_list(location):
    """DocString."""
    fList = os.listdir(location)
    for fItem in fList[:]:
        if not(fItem.endswith(".jpg")):
            fList.remove(fItem)
    return fList


def test():
    # tracker = Tracker(10, 0)
    tracker = Tracker(10, 5, 0)
    segmentor = MaskInterface()
    # path = "C:/Users/Nedsh/Documents/CS/Project/Production/karlman_filter/vidf1_33_000_f001.jpg"
    # img = cv2.imread(path)
    #
    # contours = segmentor.segment_image(path)
    # centers = []
    # x_list = []
    # y_list = []
    # for contour in contours:
    #     # Find center of mass of contours
    #     x_list = [vertex[0] for vertex in contour]
    #     y_list = [vertex[1] for vertex in contour]
    #     n_vertex = len(contour)
    #     x = int(sum(x_list) / n_vertex)
    #     y = int(sum(y_list) / n_vertex)
    #     print(x, y)
    #     centers.append([x, y])
    #     cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
    #
    # if len(contours) > 0:
    #     tracker.Update(centers)
    # for track in tracker.tracks:
    #     print("Last res:")
    #     print(track.KF.lastResult)
    #     cv2.circle(img, (int(track.KF.lastResult[0][0]),
    #                int(track.KF.lastResult[0][1])), 1,
    #                (0, 0, 255), -1)

    # path = "C:/Users/Nedsh/Documents/CS/Project/Production/karlman_filter/vidf1_33_000_f003.jpg"
    for path in get_jpg_list(IN_DIR):
        path = os.path.join(IN_DIR, path)
        img2 = cv2.imread(path)
        contours = segmentor.segment_image(path)
        centers = []
        x_list = []
        y_list = []

        for contour in contours:
            # Find center of mass of contours
            cnt = np.array(contour, dtype=np.int32)
            cnt = cnt.reshape((-1,1,2))
            cv2.polylines(img2, [cnt], True, (0,255,255))
            _, radius = cv2.minEnclosingCircle(cnt)

            if radius < 8 or radius > 20:
                continue

            x_list = [vertex[0] for vertex in contour]
            y_list = [vertex[1] for vertex in contour]
            n_vertex = len(contour)
            x = int(sum(x_list) / n_vertex)
            y = int(sum(y_list) / n_vertex)
            print(x, y)
            # cv2.circle(img2, (x, y), int(radius), (0, 255, 0), -1)
            centers.append([x, y])
            # cv2.circle(img2, (x, y), 1, (255, 0, 0), -1)
        print(centers)
        if len(contours) > 0:
            tracker.Update(centers)
        print("N Contours:", len(contours))
        print("N tracks:", len(tracker.tracks))
        for track in tracker.tracks:
            # print(track.prediction)
            print("Last res KF:")
            print(track.KF.lastResult)
            if len(track.center_history) == 0:
                print("broken")
                print(track.prediction)
                cv2.circle(img2, (int(track.KF.lastResult[0][0]),
                           int(track.KF.lastResult[0][1])), 1,
                           (0, 0, 255), -1)
                break
            print("Last res detection:")
            print(track.center_history[-1])

            cv2.circle(img2, (int(track.KF.lastResult[0][0]),
                       int(track.KF.lastResult[0][1])), 1,
                       (0, 0, 255), -1)
            cv2.circle(img2, (int(track.center[0]),
                       int(track.center[1])), 1,
                       track.color, -1)

        # cv2.imshow("Image 1", img)
        cv2.imshow("Image 2", img2)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
