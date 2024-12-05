import argparse
import datetime
import Manager as Manager
from Group_select import Group_select
from Performance import *
import time

class SimulateExp:
    def __init__(self, cameras, arms, payoff, acc_matrix, lnvs_matrix, video_size, QP_list, res_list,
                 out_folder, update_iter_gap, segment_size, pool_size, batchSize=1, train_iter = 50, test_iter=300, alias='time'):
        self.cameras = cameras
        self.all_arms = arms
        self.acc_matrix = acc_matrix
        self.lnvs_matrix = lnvs_matrix
        self.video_size = video_size
        self.QP_list = QP_list
        self.res_list = res_list
        self.update_iter_gap = update_iter_gap
        self.payoff = payoff
        self.segment_size = segment_size
        self.out_folder = out_folder
        self.batchSize = batchSize
        self.poolArticleSize = pool_size
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.alias = alias
        self.theta0 = None

    def initialize_offline(self):
        begin = time.time()
        self.OptimalArticle_list = {}
        self.regulateArticlePool()
        for camera_index in self.cameras.keys():
            Optimal_Reward, OptimalArticle = self.getOptimalReward_topk(camera_index, self.armPool, self.train_iter, 5)
            self.OptimalArticle_list[camera_index] = OptimalArticle
        print(time.time()-begin)
        return self.OptimalArticle_list, 5

    def get_acc_feature(self, qp, res, ln_knobs_video_size):
        acc_fv = np.array([
            qp / 60,
            qp ** 2 / 60 / 60,
            res,
            np.log(res * 10) / 10,
            np.log(res * 10) / 10 * qp / 60,
            ln_knobs_video_size,
        ])
        return acc_fv

    def get_video_size_feature(self, qp, res):
        return np.array([
            qp / 60 / 2,
            res,
            np.log(res * 10) / 10,
            1,
        ])

    def initialize_fit(self):
        lnvs_combined_feature_lists = []
        lnvs_response_lists = []
        acc_combined_feature_lists = []
        acc_response_lists = []
        num_segments = self.acc_matrix.shape[2]
        train_indices = slice(0, self.train_iter)
        test_indices = slice(self.train_iter, self.test_iter)
        acc_test = self.acc_matrix[..., test_indices]
        knobs_video_size_test = self.lnvs_matrix[..., test_indices]
        acc_train = np.mean(self.acc_matrix[..., train_indices], axis=-1)
        knobs_video_size_train = np.mean(self.lnvs_matrix[..., train_indices], axis=-1)
        for i in range(self.acc_matrix.shape[0]):
            lnvs_response_list = []
            lnvs_combined_feature_list = []
            acc_response_list = []
            acc_combined_feature_list = []
            t1 = len(self.res_list)
            for qp_index, qp in enumerate(self.QP_list):
                for res_index, res in enumerate(self.res_list):
                    lnvs_response_list.append(knobs_video_size_train[i, qp_index * t1 + res_index])
                    lnvs_feature_vector = self.get_video_size_feature(qp, res)
                    lnvs_combined_feature_list.append(lnvs_feature_vector)
                    acc_response_list.append(acc_train[i, qp_index * t1 + res_index])
                    acc_combined_feature_list.append(
                        self.get_acc_feature(qp, res, knobs_video_size_train[i, qp_index * t1 + res_index]))
            lnvs_combined_feature_lists.append(lnvs_combined_feature_list)
            lnvs_response_lists.append(lnvs_response_list)
            acc_combined_feature_lists.append(acc_combined_feature_list)
            acc_response_lists.append(acc_response_list)
        self.lnvs_X_train = np.array(lnvs_combined_feature_lists)
        self.lnvs_y_train = np.array(lnvs_response_lists)
        self.acc_X_train = np.array(acc_combined_feature_lists)
        self.acc_y_train = np.array(acc_response_lists)
        return self.lnvs_X_train, self.lnvs_y_train, self.acc_X_train, self.acc_y_train


    def getRealReward(self, camera_index, arm_index, iter_index):
        if arm_index >= self.payoff.shape[1]:
            print(f"Error: arm_index {arm_index} is out of bounds. Max allowed: {self.acc_matrix.shape[1] - 1}")
        return self.payoff[camera_index, arm_index, iter_index], self.acc_matrix[camera_index, arm_index, iter_index], \
        self.lnvs_matrix[camera_index, arm_index, iter_index]

    def learnRewards(self, camera_index, iter_index):
        return self.payoff[camera_index, :, iter_index], \
            self.video_size[camera_index, iter_index], \
            self.acc_matrix[camera_index, :, iter_index], \
            self.lnvs_matrix[camera_index, :, iter_index]

    def regulateArticlePool(self):
        """
         arm  pool。
        """
        all_index = range(len(self.all_arms))
        selected_pool_index = np.random.choice(all_index, self.poolArticleSize, replace=False)
        self.armPool = {}
        for si in selected_pool_index:
            self.armPool[si] = self.all_arms[si]
        if len(self.armPool) != self.poolArticleSize:
            raise AssertionError("Pool size mismatch!")

    def getOptimalReward(self, camera_index, article_pool,iter_):
        maxpayoff = float('-inf')
        best_arm = None
        best_bandwidth = 0
        for arm_index in article_pool.keys():
            payoff, acc, lnvs = self.getRealReward(camera_index, arm_index,iter_)
            if payoff > maxpayoff:
                best_arm = arm_index
                maxpayoff = payoff
                best_bandwidth = np.exp(lnvs)
        if best_arm is None:
            raise AssertionError("No optimal arm found!")
        return maxpayoff, best_bandwidth, article_pool[best_arm]

    def getOptimalReward_topk(self, camera_index, article_pool, train_iter, top_k):
        reward_sums = np.zeros(len(article_pool))
        article_indices = list(article_pool.keys())
        for i in range(train_iter):
            for idx, arm_index in enumerate(article_indices):
                payoff, _, _ = self.getRealReward(camera_index, arm_index, i)
                reward_sums[idx] += payoff
        average_rewards = [(arm_index, reward_sums[idx] / train_iter) for idx, arm_index in enumerate(article_indices)]
        average_rewards.sort(key=lambda x: x[1], reverse=True)
        top_k_arms = [article_pool[arm[0]] for arm in average_rewards[:top_k]]
        if not top_k_arms:
            raise ValueError("No optimal arms found in the top-K selection!")
        return average_rewards[0][1], top_k_arms

    def simulationAtTimeSlot(self, iter_, algorithms, camera_index):
        """
        。

        :param iter_: 
        :param algorithms: 
        :return: 、、
        """
        RegretMatrix = np.zeros(len(algorithms))
        PayoffMatrix = np.zeros(len(algorithms))
        AccuracyMatrix = np.zeros(len(algorithms))
        BandwidthMatrix = np.zeros(len(algorithms))
        TimeMatrix = np.zeros(len(algorithms))
        # 
        Optimal_payoff, Optimal_bandwidth, OptimalArticle = self.getOptimalReward(camera_index, self.armPool, iter_)

        for alg_idx, (algname, alg) in enumerate(algorithms.items()):
            start_time = time.time()

            if algname == 'Oracle':
                RegretMatrix[alg_idx] = 0
                PayoffMatrix[alg_idx] = Optimal_payoff
                BandwidthMatrix[alg_idx] = Optimal_bandwidth
                continue

            # ，
            pickedArticle = alg.decide(self.armPool, camera_index)
            payoff, acc, lnvs = self.getRealReward(camera_index, pickedArticle.aid, iter_)

            # 
            alg.small_updateParameters(pickedArticle, lnvs, camera_index, iter_)

            # 
            if (iter_ + 1) % self.update_iter_gap == 0:
                pay_off_tmp, vs_tmp, acc_tmp, lnvs_tmp = self.learnRewards(camera_index, iter_)
                alg.updateParameters(pay_off_tmp, vs_tmp, acc_tmp, lnvs_tmp, camera_index)

            # 、、、
            regret = Optimal_payoff - payoff
            RegretMatrix[alg_idx] = regret
            PayoffMatrix[alg_idx] = payoff
            AccuracyMatrix[alg_idx] = acc
            BandwidthMatrix[alg_idx] = np.exp(lnvs)

            end_time = time.time()
            # 
            TimeMatrix[alg_idx] = end_time - start_time

        return RegretMatrix, PayoffMatrix, AccuracyMatrix, BandwidthMatrix, TimeMatrix

    def runAlgorithms(self, algorithms):
        """
        ，，、、
        """
        if self.alias == 'time':
            self.starttime = datetime.datetime.now()
            timeRun = self.starttime.strftime('_%m_%d_%H_%M')
            self.alias = timeRun
        out_npz_file = os.path.join(self.out_folder, f"Results{self.alias}.npz")
        num_algorithms = len(algorithms)
        num_iterations = self.test_iter - self.train_iter
        num_cameras = len(self.cameras)

        results = {
            "Regret": np.zeros((num_cameras, num_iterations, num_algorithms)),
            "Accuracy": np.zeros((num_cameras, num_iterations, num_algorithms)),
            "Bandwidth": np.zeros((num_cameras, num_iterations, num_algorithms)),
            "CameraPayoff": np.zeros((num_cameras, num_iterations, num_algorithms)),
            "Time": np.zeros((num_cameras, num_iterations, num_algorithms)),
        }

        print(
            f"[runAlgorithms] Training iterations: {self.test_iter}, Cameras: {num_cameras}, Algorithms: {num_algorithms}")

        # 
        for iter_ in range(self.train_iter, self.test_iter):
            self.regulateArticlePool()

            all_camera_results = {
                camera_index: self.simulationAtTimeSlot(iter_, algorithms, camera_index)
                for camera_index in self.cameras.keys()
            }

            # 
            for camera_idx, (RegretMatrix, PayoffMatrix, AccuracyMatrix, BandwidthMatrix, TimeMatrix) in all_camera_results.items():
                for alg_idx, alg_name in enumerate(algorithms.keys()):
                    results["CameraPayoff"][camera_idx, iter_ - self.train_iter, alg_idx] = PayoffMatrix[alg_idx]
                    results["Regret"][camera_idx, iter_ - self.train_iter, alg_idx] = RegretMatrix[alg_idx]
                    results["Accuracy"][camera_idx, iter_ - self.train_iter, alg_idx] = AccuracyMatrix[alg_idx]
                    results["Bandwidth"][camera_idx, iter_ - self.train_iter, alg_idx] = BandwidthMatrix[alg_idx]
                    results["Time"][camera_idx, iter_ - self.train_iter, alg_idx] = TimeMatrix[alg_idx]
        average_regret = np.mean(results["Regret"], axis=(0, 1))
        average_time = np.sum(results["Time"], axis=1)
        average_time = np.average(average_time, axis=0)
        print("\nAverage Regret for each Algorithm:")
        for alg_idx, alg_name in enumerate(algorithms.keys()):
            print(f"{alg_name}: {average_regret[alg_idx]:.4f}")
        print("\nAverage Time for each Algorithm:")
        for alg_idx, alg_name in enumerate(algorithms.keys()):
            print(f"{alg_name}: {average_time[alg_idx]:.4f}")
        np.savez(out_npz_file, **results)
        print("All results saved successfully.")


def pay_off(acc, video_size, bandwidth, eta, payoff_method):
    if  payoff_method == 0:
        price = eta * video_size / bandwidth
    elif payoff_method == 1:
        price = eta * np.maximum(video_size / bandwidth - 1, 0)
    payoff = acc - price
    return payoff, price

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--in_folder', dest='in_folder', help='input the folder containing input files',default = "input_data/syn_L_50")
    parser.add_argument('--out_folder', dest='out_folder', help='input the folder to output', default  = "out_debug")
    parser.add_argument('--poolSize', dest='poolSize', type=int, help='poolSize of each iteration', default= 40)
    parser.add_argument('--seedIndex', dest='seedIndex', type=int, help='seedIndex',default=1)
    args = parser.parse_args()
    acc = np.load(f"data/experiment_knobs_segment_acc.npy")
    segment_size = 1
    num_segments = acc.shape[-2] // segment_size
    new_shape = acc.shape[:-2] + (num_segments, segment_size, acc.shape[-1])
    reward_matrix = acc.reshape(new_shape)
    reward_matrix = np.sum(reward_matrix, axis=-2)
    save_label = 0
    acc = element2result(reward_matrix)[..., save_label]
    motion_feature_matrix = np.load(r"E:\dataset\dash_video\combined_features.npy")
    knobs_video_size = np.load("data/experiment_video_sizes_knobs_segments.npy")
    video_size = np.load("data/experiment_video_sizes_segments.npy")
    new_shape = knobs_video_size.shape[:-1] + (num_segments, segment_size)
    knobs_video_size = knobs_video_size.reshape(new_shape)
    knobs_video_size = np.mean(knobs_video_size, axis=-1)
    knobs_video_size = knobs_video_size * 8 / 1024
    mean_value = np.mean(knobs_video_size, axis=-1)
    mean_value = np.mean(mean_value, axis=0)
    qp_begin = 0
    qp_end = 12
    res_begin = 0
    res_end = 4
    QP_list = [26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
    res_list = [1, 0.7, 0.5, 0.3]
    QP_list = QP_list[qp_begin:qp_end]
    res_list = res_list[res_begin:res_end]
    acc = acc[:,qp_begin:qp_end,res_begin:res_end,:]
    knobs_video_size = knobs_video_size[:,qp_begin:qp_end,res_begin:res_end,:]
    mean_value = np.mean(knobs_video_size, axis=-1)
    mean_value = np.mean(mean_value, axis=0)
    new_shape = video_size.shape[:-1] + (num_segments, segment_size)
    video_size = video_size.reshape(new_shape)
    video_size = np.mean(video_size, axis=-1)
    video_size = video_size * 8 /1024
    bandwidth  = 200
    eta = 1
    payoff_method = 1
    payoff, latency = pay_off(acc, knobs_video_size, bandwidth, eta, payoff_method)
    lnvs_matrix = np.log(knobs_video_size)
    a, b, c, d = lnvs_matrix.shape
    lnvs_matrix = lnvs_matrix.reshape(a, b * c, d)
    acc_matrix = acc.reshape(a, b * c, d)
    payoff = payoff.reshape(a, b * c, d)
    camera_num = 43
    begin_camera = 0
    dim = 4
    UM = Manager.CameraManager(begin_camera,camera_num)
    UM.loadCameras()
    AM = Manager.ArmManager()
    AM.loadArms(QP_list,res_list)
    out_folder = "output_directory"
    test_iter = 300//segment_size
    train_iter = 50//segment_size
    update_iter_gap = 50/segment_size
    save_name = "time"
    simExperiment = SimulateExp(UM.cameras, AM.arms, payoff, acc_matrix, lnvs_matrix, video_size, QP_list, res_list,
                                out_folder, update_iter_gap, segment_size, pool_size = len(QP_list)*len(res_list),
                                train_iter = train_iter, test_iter=test_iter, alias = save_name)
    vs_matrix_features,vs_matrix,acc_matrix_features,acc_matrix = simExperiment.initialize_fit()
    paras = {'lambda': 1, 'beta': 1, 'alpha': 1}
    Group_select = Group_select(begin_camera, camera_num, dim, bandwidth, eta, payoff_method, para=paras)
    Group_select.offline_learn(vs_matrix_features,vs_matrix,acc_matrix)
    algorithms = {
        'Group_select': Group_select,
    }
    simExperiment.runAlgorithms(algorithms)

