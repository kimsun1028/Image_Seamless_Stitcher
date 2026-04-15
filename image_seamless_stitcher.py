import numpy as np
import cv2 as cv


def resize_to_fit(img, max_width=1600, max_height=900):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        return cv.resize(img, new_size, interpolation=cv.INTER_AREA)
    return img


def stitch_with_homography(base_img, warp_img, H):
    h1, w1 = base_img.shape[:2]
    h2, w2 = warp_img.shape[:2]

    corners_base = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32).reshape(-1, 1, 2)
    corners_warp = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32).reshape(-1, 1, 2)

    warped_corners = cv.perspectiveTransform(corners_warp, H)
    all_corners = np.concatenate([corners_base, warped_corners], axis=0)

    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    tx, ty = -x_min, -y_min
    out_w, out_h = x_max - x_min, y_max - y_min

    translation = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    stitched = cv.warpPerspective(warp_img, translation @ H, (out_w, out_h))
    stitched[ty:ty + h1, tx:tx + w1] = base_img
    return stitched

# Load two images
img1 = cv.imread('data/image01.jpg')
img2 = cv.imread('data/image02.jpg')
assert (img1 is not None) and (img2 is not None), 'Cannot read the given images'

# Retrieve matching points
fdetector = cv.BRISK_create()
keypoints1, descriptors1 = fdetector.detectAndCompute(img1, None)
keypoints2, descriptors2 = fdetector.detectAndCompute(img2, None)

fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
match = fmatcher.match(descriptors1, descriptors2)

# Calculate planar homography and merge two images
pts1, pts2 = [], []
for i in range(len(match)):
    pts1.append(keypoints1[match[i].queryIdx].pt)
    pts2.append(keypoints2[match[i].trainIdx].pt)
pts1 = np.array(pts1, dtype=np.float32)
pts2 = np.array(pts2, dtype=np.float32)

H, inlier_mask = cv.findHomography(pts2, pts1, cv.RANSAC)
img_merged = stitch_with_homography(img1, img2, H)
cv.imwrite('data/stitched_result.jpg', img_merged)

# Show the merged image
img_matched = cv.drawMatches(img1, keypoints1, img2, keypoints2, match, None, None, None,
                             matchesMask=inlier_mask.ravel().tolist()) # Remove `matchesMask` if you want to show all putative matches
top = np.hstack((img1, img2))

# Normalize panel widths for vertical stacking.
panel_width = max(top.shape[1], img_matched.shape[1], img_merged.shape[1])

def pad_to_width(img, width):
    h, w = img.shape[:2]
    if w == width:
        return img
    pad = np.zeros((h, width - w, 3), dtype=img.dtype)
    return np.hstack((img, pad))

merge = np.vstack((pad_to_width(top, panel_width),
                   pad_to_width(img_matched, panel_width),
                   pad_to_width(img_merged, panel_width)))
merge = resize_to_fit(merge)
cv.imshow('Planar Image Stitching', merge)
cv.waitKey(0)
cv.destroyAllWindows()