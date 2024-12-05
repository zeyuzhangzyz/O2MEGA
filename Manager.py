import numpy as np

class Arm:
    def __init__(self, aid, qp, resolution):
        self.aid = aid
        self.qp = qp
        self.res = resolution
        self.acc_fv = None
        self.vs_fv = None

    def get_acc_feature(self, ln_knobs_video_size):
        self.acc_fv = np.array([[
            self.qp / 60,
            self.qp ** 2 / 60 / 60,
            self.res,
            np.log(self.res * 10) / 10,
            np.log(self.res * 10) / 10 * self.qp / 60,
            ln_knobs_video_size,
        ]])
        return self.acc_fv

    def get_video_size_feature(self):
        return np.array([[
            self.qp / 60 / 2,
            self.res,
            np.log(self.res * 10) / 10,
            1,
        ]])

class ArmManager:
    def __init__(self):
        self.arms = {}
        self.n_arms = 0
        self.acc_dim = 6
        self.video_size_dim = 4
    def loadArms(self,qp_list,res_list):
        arm_id = 0
        vs_feature_vectors = []
        for qp in qp_list:
            for res in res_list:
                arm = Arm(arm_id, qp, res)
                vs_feature_vectors.append(arm.get_video_size_feature())
                self.arms[arm_id] = arm
                arm_id += 1
        self.n_arms = len(self.arms)
        feature_matrix = np.vstack(vs_feature_vectors)
        for i, arm_id in enumerate(self.arms):
            self.arms[arm_id].vs_fv = feature_matrix[i].reshape(4, 1)

class Camera:
    def __init__(self, uid):
        self.uid = uid

class CameraManager:
    def __init__(self, begin, num_cameras):
        self.cameras = {}
        self.num_cameras = num_cameras
        self.begin = begin
    def loadCameras(self):
        for uid in range(self.begin, self.begin+self.num_cameras):
            self.cameras[uid] = Camera(uid)

