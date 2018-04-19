from segmentation.segmentation import MaskInterface
from .tracker import Tracker
from scipy.ndimage.measurements import center_of_mass
import cv2

def test():
    tracker = Tracker(10, 0)
    segmentor = MaskInterface()
    # path = "C:/Users/Nedsh/Documents/CS/Project/Production/karlman_filter/vidf1_33_000_f002.jpg"
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

    path = "C:/Users/Nedsh/Documents/CS/Project/Production/karlman_filter/vidf1_33_000_f003.jpg"
    img2 = cv2.imread(path)
    contours = segmentor.segment_image(path)
    centers = []
    x_list = []
    y_list = []
    for contour in contours:
        # Find center of mass of contours
        x_list = [vertex[0] for vertex in contour]
        y_list = [vertex[1] for vertex in contour]
        n_vertex = len(contour)
        x = int(sum(x_list) / n_vertex)
        y = int(sum(y_list) / n_vertex)
        print(x, y)
        centers.append([x, y])
        cv2.circle(img2, (x, y), 1, (255, 0, 0), -1)
    print(centers)
    if len(contours) > 0:
        tracker.Update(centers)
    for track in tracker.tracks:
        print(track.prediction)
        print("Last res KF:")
        print(track.KF.lastResult)
        print("Last res detection:")
        print(track.center_history[-1])

        cv2.circle(img2, (int(track.center_history[-1][0][0]),
                   int(track.center_history[-1][0][1])), 1,
                   (0, 0, 255), -1)

    # cv2.imshow("Image 1", img)
    cv2.imshow("Image 2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
