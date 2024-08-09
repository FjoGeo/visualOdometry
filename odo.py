import os
import numpy as np
import cv2
from tqdm import tqdm

from lib.visualization import plotting

class VisualOdometry():
    def __init__(self, data_dir):
        self.K = np.array([
            [918.5995483398438, 0.0, 641.582275390625],
            [0.0, 918.8787841796875, 379.2149963378906],
            [0.0, 0.0, 1.0]
        ])
        
        self.P = np.array([
            [918.5995483398438, 0.0, 641.582275390625, 0.0],
            [0.0, 918.8787841796875, 379.2149963378906, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        self.images = self._load_images(os.path.join(data_dir,"image_l"))
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=500)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)


    @staticmethod
    def _load_calib(filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]

        return K, P

    @staticmethod
    def _load_images(filepath):
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
    

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    

    def get_matches(self, i):
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
        kp2, des2 = self.orb.detectAndCompute(self.images[i], None)
        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Find the matches there do not have a too high distance
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        # If not enough good matches, return None
        if len(good) < 8:
            return None, None
        
        # Drawing
        draw_params = dict(matchColor = -1, # draw matches in green color
            singlePointColor = None,
            matchesMask = None, # draw only inliers
            flags = 2)

        # img3 = cv2.drawMatches(self.images[i], kp1, self.images[i-1],kp2, good ,None,**draw_params)
        # cv2.imshow("image", img3)
        # cv2.waitKey(200)

        # Get the image points from the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2
    

    def get_pose(self, q1, q2):
        # Essential matrix
        E, mask = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        # If not enough inliers, return None
        if E is None or np.sum(mask) < 8:
            return None

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2)

        if R is None or t is None:
            return None

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    

    def decomp_essential_mat(self, E, q1, q2):
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]
    


def main():
    data_dir = "amb_sequences"
    vo = VisualOdometry(data_dir)
 
    estimated_path = []
    cur_pose = np.eye(4)
    for i in tqdm(range(1, len(vo.images))):
        try:
            q1, q2 = vo.get_matches(i)
            if q1 is None or q2 is None:
                continue

            transf = vo.get_pose(q1, q2)
            if transf is None or np.isnan(transf).any() or np.isinf(transf).any():
                continue

            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))

            estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        except Exception as e:
            continue
    
    # plotting.visualize_paths2(estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")
    np.savetxt('estimated_path.csv', estimated_path, delimiter=',')

if __name__ == "__main__":
    main()