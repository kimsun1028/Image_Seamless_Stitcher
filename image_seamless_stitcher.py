import numpy as np
import cv2 as cv


def resize_for_saving(img, max_width=1920, max_height=1080):
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


def estimate_homography(base_img, warp_img):
    detector = cv.BRISK_create()
    keypoints_base, descriptors_base = detector.detectAndCompute(base_img, None)
    keypoints_warp, descriptors_warp = detector.detectAndCompute(warp_img, None)

    if descriptors_base is None or descriptors_warp is None:
        raise RuntimeError('Feature extraction failed on one of the images.')

    matcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
    knn_matches = matcher.knnMatch(descriptors_warp, descriptors_base, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:
        raise RuntimeError(f'Not enough good matches: {len(good_matches)}')

    pts_warp = np.float32([keypoints_warp[m.queryIdx].pt for m in good_matches])
    pts_base = np.float32([keypoints_base[m.trainIdx].pt for m in good_matches])

    H, _ = cv.findHomography(pts_warp, pts_base, cv.RANSAC, 4.0)
    if H is None:
        raise RuntimeError('Homography estimation failed.')

    return H


def warp_corners(img, H):
    h, w = img.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    return cv.perspectiveTransform(corners, H)


def place_with_mask(dst, src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    mask = gray > 0
    dst[mask] = src[mask]


def feather_blend(images, masks):
    eps = 1e-6
    accum = np.zeros_like(images[0], dtype=np.float32)
    weight_sum = np.zeros(masks[0].shape, dtype=np.float32)

    for img, mask in zip(images, masks):
        dist = cv.distanceTransform(mask, cv.DIST_L2, 3)
        weight = np.where(mask > 0, dist + eps, 0).astype(np.float32)

        accum += img.astype(np.float32) * weight[..., None]
        weight_sum += weight

    result = np.zeros_like(images[0], dtype=np.uint8)
    valid = weight_sum > eps
    result[valid] = (accum[valid] / weight_sum[valid, None]).clip(0, 255).astype(np.uint8)
    return result


def preserve_center_sharpness(merged, center_img, x, y, edge_width=120):
    h, w = center_img.shape[:2]
    roi = merged[y:y + h, x:x + w].astype(np.float32)
    center_f = center_img.astype(np.float32)

    yy, xx = np.ogrid[:h, :w]
    dist_x = np.minimum(xx, w - 1 - xx)
    dist_y = np.minimum(yy, h - 1 - yy)
    dist_to_edge = np.minimum(dist_x, dist_y).astype(np.float32)
    alpha = np.clip(dist_to_edge / float(edge_width), 0.0, 1.0)

    blended = roi * (1.0 - alpha[..., None]) + center_f * alpha[..., None]
    merged[y:y + h, x:x + w] = blended.clip(0, 255).astype(np.uint8)
    return merged



img1 = cv.imread('data/image01.jpg')
img2 = cv.imread('data/image02.jpg')
img3 = cv.imread('data/image03.jpg')
assert (img1 is not None) and (img2 is not None) and (img3 is not None), 'Cannot read the given images'


H_right_to_center = estimate_homography(img2, img1)
H_left_to_center = estimate_homography(img2, img3)

h2, w2 = img2.shape[:2]
corners_center = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32).reshape(-1, 1, 2)
corners_right = warp_corners(img1, H_right_to_center)
corners_left = warp_corners(img3, H_left_to_center)

all_corners = np.concatenate([corners_center, corners_right, corners_left], axis=0)
x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

tx, ty = -x_min, -y_min
out_w, out_h = x_max - x_min, y_max - y_min

T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

warped_left = cv.warpPerspective(img3, T @ H_left_to_center, (out_w, out_h))
warped_right = cv.warpPerspective(img1, T @ H_right_to_center, (out_w, out_h))
warped_center = np.zeros((out_h, out_w, 3), dtype=np.uint8)
warped_center[ty:ty + h2, tx:tx + w2] = img2

mask_left = cv.warpPerspective(np.ones(img3.shape[:2], dtype=np.uint8) * 255, T @ H_left_to_center, (out_w, out_h))
mask_right = cv.warpPerspective(np.ones(img1.shape[:2], dtype=np.uint8) * 255, T @ H_right_to_center, (out_w, out_h))
mask_center = np.zeros((out_h, out_w), dtype=np.uint8)
mask_center[ty:ty + h2, tx:tx + w2] = 255

img_merged = feather_blend(
    [warped_left, warped_center, warped_right],
    [mask_left, mask_center, mask_right]
)

img_merged = preserve_center_sharpness(img_merged, img2, tx, ty, edge_width=120)

img_to_save = resize_for_saving(img_merged)

output_path = 'data/stitched_result.jpg'
saved = cv.imwrite(output_path, img_to_save)
if not saved:
    raise RuntimeError(f'Failed to write output image: {output_path}')

print(f'Stitched image saved to: {output_path}')

img_to_show = resize_for_saving(img_merged, 1600, 900)
cv.imshow('Stitched Image', img_to_show)
cv.waitKey(0)
cv.destroyAllWindows()
