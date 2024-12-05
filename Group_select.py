import numpy as np
import networkx as nx

class Camera:
    """Represents an individual camera with LinUCB capabilities."""
    def __init__(self, dim, bandwidth, eta, lambda_, cameraid, init='zero', beta=1.0, alpha=1.0, payoff_method = 0):
        self.dim = dim
        self.bandwidth = bandwidth
        self.eta = eta
        self.lambda_ = lambda_
        self.beta = beta
        self.alpha = alpha
        self.time = 1
        self.A = lambda_ * np.eye(dim)
        self.A_inv = np.linalg.inv(self.A)
        self.b = np.zeros((dim, 1))
        self.theta = np.random.rand(dim) if init != 'zero' else np.zeros(dim)
        self.acc_matrix = None
        self.cameraid = cameraid
        self.payoff_method = payoff_method
    def estimate_reward(self, features):
        mean = np.dot(self.theta.T, features)
        var = np.sqrt(np.dot(features.T, np.dot(self.A_inv, features)))
        pta = mean + self.alpha * var
        return pta, mean, self.alpha, var

    def calculate_payoff(self, acc, video_size):
        price = self.eta * video_size / self.bandwidth
        return acc - price

    # def calculate_payoff(self, acc, video_size):
    #     price = self.alpha * np.maximum(video_size / self.bandwidth - 1, 0)
    #     payoff = acc - price
    #     return payoff


    def update(self, features, reward):
        gamma = 1
        self.A += np.outer(features, features)*gamma
        self.b += np.outer(features, reward)*gamma
        self.A_inv = np.linalg.inv(self.A)
        self.theta = np.dot(self.A_inv, self.b)
        self.time += 1


    def offline_update(self, features, reward):
        self.A += np.outer(features, features)
        self.b += np.outer(features, reward)
        self.A_inv = np.linalg.inv(self.A)
        self.theta = np.dot(self.A_inv, self.b)
        self.time += 1

class GroupStruct:
    """Represents a group of cameras with aggregated parameters."""
    def __init__(self, dim, bandwidth, eta, lambda_, init='zero', beta=1.0, alpha=1.0):
        self.dim = dim
        self.bandwidth = bandwidth
        self.eta = eta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.beta = beta
        self.A = lambda_ * np.eye(dim)
        self.b = np.zeros((dim, 1))
        self.A_inv = np.linalg.inv(self.A)
        self.theta = np.zeros(dim)
        self.cameras = []

    def add_camera(self, camera):
        """Adds a camera to the group."""
        self.cameras.append(camera)
        self.A += camera.A - self.lambda_ * np.eye(self.dim)
        self.b += camera.b
        self.A_inv = np.linalg.inv(self.A)
        self.theta = np.dot(self.A_inv, self.b)

    def update(self, features, reward):
        self.A += np.outer(features, features)
        self.b += np.outer(features, reward)
        self.A_inv = np.linalg.inv(self.A)
        # self.A_inv = np.linalg.solve(self.A, np.eye(self.dim))
        self.theta = np.dot(self.A_inv, self.b)
        # self.theta = np.linalg.solve(self.A, self.b)
    def aggregate_group_parameters(self):
        """Aggregates parameters across all cameras in the group."""
        self.A_inv = np.linalg.inv(self.A)
        # self.A_inv = np.linalg.solve(self.A, np.eye(self.dim))
        self.theta = np.dot(self.A_inv, self.b)
        # self.theta = np.linalg.solve(self.A, self.b)

class Group_select:
    """Orchestrates the system, managing cameras and groups."""
    def __init__(self, begin_camera, camera_num, dim, bandwidth, eta, payoff_method, para, init='zero'):
        self.dim = dim
        self.lambda_ = para['lambda']
        self.bandwidth = bandwidth
        self.eta = eta
        self.beta = para['beta']
        self.alpha = para.get('alpha', -1)
        self.cameras = {}  # Individual cameras
        self.camera_group = {}  # Maps cameras to groups
        self.groups = {}  # Groups of cameras
        self.graph = nx.complete_graph(range(begin_camera, begin_camera + camera_num))
        self.payoff_method = payoff_method
        # Initialize cameras
        for cameraid in range(begin_camera, begin_camera + camera_num):
            self.cameras[cameraid] = Camera(dim, bandwidth, eta, self.lambda_, cameraid, init, self.beta, self.alpha, self.payoff_method)

        # Initialize single group
        self.groups[0] = GroupStruct(dim, bandwidth, eta, self.lambda_, init, self.beta, self.alpha)
        for cameraid in self.cameras:
            self.groups[0].add_camera(self.cameras[cameraid])
            self.camera_group[cameraid] = 0

    def should_split(self, camera1, camera2):
        """Determines if two cameras should be split into different groups."""
        u1, u2 = self.cameras[camera1], self.cameras[camera2]
        factor_t1 = np.sqrt((1 + np.log(1 + u1.time)) / (1 + u1.time))
        factor_t2 = np.sqrt((1 + np.log(1 + u2.time)) / (1 + u2.time))
        return np.linalg.norm(u1.theta - u2.theta) > self.beta/2 * (factor_t1 + factor_t2)

    def update_groups(self, current_group):
        """Updates groups by examining intra-group edges and splitting if necessary."""
        # Create a subgraph for the current group
        current_group.aggregate_group_parameters()
        group_nodes = [camera.cameraid for camera in current_group.cameras]
        if len(group_nodes) == 1:
            return
        subgraph = self.graph.subgraph(group_nodes).copy()  # Copy to modify safely
        # Identify edges to remove within the group
        edges_to_remove = [
            edge for edge in subgraph.edges()
            if self.should_split(edge[0], edge[1])
        ]
        update = False
        if len(edges_to_remove) == 0:
            update = True
        self.graph.remove_edges_from(edges_to_remove)
        # Find connected components in the updated subgraph
        connected_components = list(nx.connected_components(self.graph))

        # Handle group splits or retention
        if len(connected_components) > 1 and update:
            # Group has split into multiple connected components
            new_groups = {}
            next_group_id = 0
            for component in connected_components:  # Ignore components with only 1 camera, those should remain in the original group
                new_group = GroupStruct(self.dim, self.bandwidth, self.eta, self.lambda_)

                for cameraid_in_component in component:
                    camera = self.cameras[cameraid_in_component]
                    new_group.add_camera(camera)
                    self.camera_group[cameraid_in_component] = next_group_id  # Update camera's group mapping

                new_groups[next_group_id] = new_group
                next_group_id += 1

            # Update groups with the new set of groups, including the original
            self.groups = new_groups

    def decide(self, pool_arms, cameraid):
        """Selects the best arm for a camera within its group."""
        group_id = self.camera_group[cameraid]
        group = self.groups[group_id]
        max_reward = float('-inf')
        best_arm = None
        for arm in pool_arms.values():
            fv = arm.vs_fv
            mean = np.dot(group.theta.T, fv)
            var = np.sqrt(np.dot(fv.T, np.dot(group.A_inv, fv)))
            reward = self.cameras[cameraid].calculate_payoff(self.cameras[cameraid].acc_matrix[arm.aid], np.exp(mean)) + self.alpha * var
            if reward > max_reward:
                max_reward = reward
                best_arm = arm
        # print(f"Camera {cameraid} selected arm {best_arm.aid} with reward {max_reward}")
        return best_arm

    def small_updateParameters(self, arm, reward, cameraid, iter_):
        """Updates parameters for a specific camera and refreshes groups."""
        self.cameras[cameraid].update(arm.vs_fv, reward)
        current_group_id = self.camera_group[cameraid]
        current_group = self.groups[current_group_id]
        current_group.update(arm.vs_fv, reward)
        self.update_groups(current_group)

    def updateParameters(self, pay_off_tmp, vs_tmp, acc_tmp, lnvs_tmp, cameraid):
        """Updates parameters for a specific camera and refreshes groups."""
        pass

    def offline_learn(self, vs_matrix_features, vs_matrix, acc_matrix, train_rounds = 50):
        """Trains the system in an offline manner."""
        for aid in range(acc_matrix.shape[1]):
            for cameraid, camera in self.cameras.items():
                camera.acc_matrix = acc_matrix[cameraid]
                camera.update(vs_matrix_features[cameraid,aid]*train_rounds, vs_matrix[cameraid,aid]*train_rounds)
                current_group_id = self.camera_group[cameraid]
                current_group = self.groups[current_group_id]
                current_group.update(vs_matrix_features[cameraid,aid]*train_rounds, vs_matrix[cameraid,aid]*train_rounds)
            self.update_groups(current_group)
        # print(self.groups)