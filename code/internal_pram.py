"""
Copyright 2023 ground0state
Released under the MIT license
"""
from os.path import join

import sys
import cv2
import numpy as np
from scipy.linalg import svd


def get_sift_keypoints(img1, img2, n_matches=None):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)[:n_matches]
    return keypoints1, keypoints2, matches


def match_filter(keypoints1, keypoints2, matches):
    match_points1, match_points2 = [], []
    for i_match in matches:
        match_points1.append(keypoints1[i_match.queryIdx].pt)
        match_points2.append(keypoints2[i_match.trainIdx].pt)
    match_points1 = np.array(match_points1).astype(np.float64)
    match_points2 = np.array(match_points2).astype(np.float64)
    return match_points1, match_points2, matches


def normalize_points(homo_points):
    """ 画像座標の1,2次元成分を正規化する. """
    mean = homo_points[:2].mean(axis=1)
    scale = np.sqrt(2) / np.mean(
        np.linalg.norm(
            homo_points[:2] - mean.reshape(2, 1), axis=0))

    mean = np.append(mean, 0)
    T = np.array([[scale, 0, -scale*mean[0]],
                  [0, scale, -scale*mean[1]],
                  [0, 0, 1]])

    homo_points = T @ homo_points
    return homo_points, T


def find_fundamental_matrix(points1, points2, verbose=0):
    """ 正規化8点アルゴリズムでF行列を推定する. """

    assert points1.shape[1] == points2.shape[1] == 2
    assert points1.shape[0] >= 8

    points1 = np.array(
        [(kpt[0], kpt[1], 1) for kpt in points1]).T.astype(np.float64)
    points2 = np.array(
        [(kpt[0], kpt[1], 1) for kpt in points2]).T.astype(np.float64)

    # 正規化
    points1_norm, T1 = normalize_points(points1)
    points2_norm, T2 = normalize_points(points2)

    # エピポーラ拘束式
    A = np.zeros((points1.shape[1], 9), dtype=np.float64)
    for i in range(points1.shape[1]):
        A[i] = [
            points1_norm[0, i]*points2_norm[0, i],
            points1_norm[1, i]*points2_norm[0, i],
            points2_norm[0, i],
            points1_norm[0, i]*points2_norm[1, i],
            points1_norm[1, i]*points2_norm[1, i],
            points2_norm[1, i],
            points1_norm[0, i],
            points1_norm[1, i],
            1.0
        ]

    # 特異値分解で連立方程式を解く
    U, S, Vt = svd(A, lapack_driver="gesvd")
    # S, U, Vt = cv2.SVDecomp(A.T @ A)  # OpenCVを使う場合

    if verbose >= 1:
        print("SVD decomposition eigen values:", S)
    F = Vt[-1].reshape(3, 3)

    # 最小特異値を厳密に0とし、F行列をランク2にする
    U, S, Vt = svd(F, lapack_driver="gesvd")
    # S, U, Vt = cv2.SVDecomp(F)  # OpenCVを使う場合

    if verbose >= 1:
        print("Rank SVD decomposition eigen values:", S)

    S = S.reshape(-1)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    # 正規化を戻す
    F = T2.T @ F @ T1
    # f_33を1にする
    F = F / F[2, 2]

    return F


def vec2outer(a):
    assert a.ndim == 1
    assert a.shape[0] == 3
    return np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])


def _triangulate_one_point(P1, P2, pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    X = np.vstack([
        x1*P1[2, :]-P1[0, :],
        y1*P1[2, :]-P1[1, :],
        x2*P2[2, :]-P2[0, :],
        y2*P2[2, :]-P2[1, :],
    ])
    _, _, Vt = svd(X)
    x_w = Vt[-1]
    x_w = x_w / x_w[-1]
    return x_w


def triangulate(P1, P2, points1, points2):
    assert points1.shape == points2.shape
    assert points1.ndim <= 2

    if points1.ndim == 1:
        # 2次元にする
        points1 = points1.reshape(1, -1)
        points2 = points2.reshape(1, -1)

    if points1.shape[1] == 3:
        # 同次座標の場合
        points1 = points1[:, :2]
        points2 = points2[:, :2]

    X_w = []
    for pt1, pt2 in zip(points1, points2):
        x_w = _triangulate_one_point(P1, P2, pt1, pt2)
        X_w.append(x_w)
    X_w = np.vstack(X_w)

    return X_w


def estimate_projection(F):
    U, _, _ = svd(F)
    e = U[:, -1] / U[-1, -1]
    S = vec2outer(e)

    P1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])
    P2 = np.column_stack([S @ F, e])
    return P1, P2


if __name__ == "__main__":

    # Parameters ---------------
    argv = sys.argv
    seed = 0
    data_dir = "."
    image1_name = argv[1]
    image2_name = argv[2]
    n_matches = 100
    verbose = 0
    # --------------------------

    img1 = cv2.imread(
        join(data_dir, image1_name), cv2.IMREAD_COLOR)
    img2 = cv2.imread(
        join(data_dir, image2_name), cv2.IMREAD_COLOR)

    # 画像の対応する点を求める
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    keypoints1, keypoints2, matches = get_sift_keypoints(
        img1_gray, img2_gray, n_matches)

    match_img = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, matches,
        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("match_image.jpg", match_img)

    # F行列を推定する
    match_points1, match_points2, matches = match_filter(
        keypoints1, keypoints2, matches)
    F = find_fundamental_matrix(
        match_points1, match_points2, verbose=verbose)
    F_, mask = cv2.findFundamentalMat(
        match_points1, match_points2, method=cv2.FM_8POINT)
    rmse = np.sqrt(np.mean((F - F_)**2))
    print("OpenCVとの差 RMSE:", rmse)

    P1, P2 = estimate_projection(F)
    print(P1)
    print(P2)