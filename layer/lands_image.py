def generate_landmarks_heatmap(landmarks, out_shape, patch_size):
    '''
        landmarks(array):           landmarks to generate heatmap
        out_shape(list/tuple):      shape of output heatmap in (H, W)
        patch_size(int):            landmarks heatmap patch size
    '''
    landmarks = landmarks.copy()
    num_points = len(landmarks)
    img_shape = out_shape
    half_size = patch_size // 2
    offsets = np.array(list(itertools.product(range(-half_size, half_size + 1), range(-half_size, half_size + 1))))
    mid_value = np.zeros((num_points, offsets.shape[0], offsets.shape[1]), dtype=np.float32)
    mid_value_l2 = np.zeros((num_points, offsets.shape[0]), dtype=np.float32)
    mid_value_sq = np.zeros((num_points, offsets.shape[0]), dtype=np.float32)
    max_mask = np.zeros((1, img_shape[0], img_shape[1]), int)

    landmarks[:, 0] = np.clip(landmarks[:, 0], half_size, img_shape[1] - 1 - half_size)
    landmarks[:, 1] = np.clip(landmarks[:, 1], half_size, img_shape[0] - 1 - half_size)

    imgs = np.zeros((landmarks.shape[0], 1, img_shape[0], img_shape[1]), dtype=np.float32)

    for i in range(landmarks.shape[0]):
        img = np.zeros((1, img_shape[0], img_shape[1]), dtype=np.float32)

        intLandmark = landmarks[i].astype('int32')
        locations = offsets + intLandmark
        dxdy = landmarks[i] - intLandmark

        offsetsSubPix = offsets - dxdy
        mid_value[i, :] = offsetsSubPix
        mid_value_l2[i, :] = np.sqrt(np.sum(offsetsSubPix * offsetsSubPix, axis=1) + 1e-6)
        mid_value_sq[i, :] = (1 + mid_value_l2[i, :]) * (1 + mid_value_l2[i, :])
        vals = 1 / (1 + mid_value_l2[i, :])

        img[0, locations[:, 1], locations[:, 0]] = vals
        imgs[i, :] = img.copy()

    max_mix = np.max(imgs, 0)
#     for ii in range(landmarks.shape[0]):
#         max_mask = np.where(max_mix == imgs[ii, :], ii, max_mask)
    return max_mix[0]

