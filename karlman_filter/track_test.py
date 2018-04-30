from segmentation.segmentation import MaskInterface
from .tracker_old import Tracker
from scipy.ndimage.measurements import center_of_mass
import cv2
import os
import numpy as np

# IN_DIR = "C:/Users/Nedsh/Documents/CS/Project/DatasetRaw/ucsdpeds/vidf_jpg/vidf4_33_014.y"
# IN_DIR = "C:/Users/Nedsh/Documents/CS/Project/DatasetRaw/ucsdpeds/vidf_jpg/vidf6_33_018.y"
# vidf2_33_005.y
# IN_DIR = "C:/Users/Nedsh/Documents/CS/Project/DatasetRaw/ucsdpeds/vidf_jpg/vidf2_33_005.y"
# IN_DIR = "C:/Users/Nedsh/Documents/CS/Project/DatasetRaw/ucsdpeds/vidf_jpg/vidf1_33_000.y"

IN_DIR = "./example/raw"


def get_jpg_list(location):
    """DocString."""
    fList = os.listdir(location)
    for fItem in fList[:]:
        if not(fItem.endswith(".jpg")):
            fList.remove(fItem)
    return fList


def test():
    # tracker = Tracker(10, 0)
    tracker = Tracker(5, 2, 0)
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

            # if radius < 10 or radius > 30:
            if radius < 8 or radius > 30:
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
            if len(track.center_history) > 1:
                for i in range(len(track.center_history) - 1):
                    cv2.line(img2, (int(track.center_history[i][0][0]),
                             int(track.center_history[i][0][1])),
                             (int(track.center_history[i+1][0][0]),
                             int(track.center_history[i+1][0][1])), track.color, 1)

            if len(track.center_history) > 8:
                t_list = track.center_history[-8:]
                count = 0
                avg_diff_x = 0
                avg_diff_y = 0
                for i in range(len(t_list) - 1):
                    a = t_list[i][0]
                    b = t_list[i+1][0]
                    avg_diff_x += b[0] - a[0]
                    avg_diff_y += b[1] - a[1]
                    count += 1
                avg_diff_x = avg_diff_x/count
                avg_diff_y = avg_diff_y/count
                pos_list = []
                for i in range(10):
                    diff_x = (i+1) * avg_diff_x
                    diff_y = (i+1) * avg_diff_y
                    new_x = diff_x + t_list[-1][0][0]
                    new_y = diff_y + t_list[-1][0][1]
                    pos_list.append([new_x, new_y])
                for i in range(len(pos_list) - 1):
                    cv2.line(img2, (int(pos_list[i][0]),
                             int(pos_list[i][1])),
                             (int(pos_list[i+1][0]),
                             int(pos_list[i+1][1])), (0, 0, 255), 1)

        # cv2.imshow("Image 1", img)
        cv2.imshow("Image 2", img2)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            break
        # cv2.waitKey(0)
        #
