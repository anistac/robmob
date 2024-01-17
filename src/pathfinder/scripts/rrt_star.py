"""
Class to perform RRT* path planning.
The map is a 2D occupancy grid (numpy array) with float values between 0 and 1.
0 means free space, 1 means occupied space.
"""
import cv2
import time
import math
from typing import List, Tuple
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

OCCUPANCY_THRESH = 0.8
SCALING_FACTOR = 1000
DISPLAY_RESIZE_FACTOR = 12

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
GREY = (128, 128, 128)


class RRTStar:
    def __init__(self, occupancy_grid: np.ndarray) -> None:
        self.occupancy_grid = occupancy_grid
        self.graph = nx.Graph()

    def _get_random_node(self) -> np.ndarray:
        """
        Function to get a random node in the map.
        :return: random node [x,y]
        """
        grid_size = self.occupancy_grid.shape
        x = np.random.randint(0, grid_size[0])
        y = np.random.randint(0, grid_size[1])
        return np.array([x, y])

    def _find_nearest_node(self, node: np.ndarray) -> np.ndarray:
        """
        Function to find the nearest node to the given node.
        :param node: node [x,y]
        :return: nearest node [x,y]
        """
        # convert nodes to a KDTree for fast nearest neighbor search
        node_list = list(self.graph.nodes)
        node_kd_tree = cKDTree(node_list)
        # find nearest node
        nearest_node_idx = node_kd_tree.query(node, 1)[1]
        return node_list[nearest_node_idx]

    def _steer(self, from_node: np.ndarray, to_node: np.ndarray, max_dist: float) -> np.ndarray:
        """
        Function to steer from one node to another node.
        :param from_node: from node [x,y]
        :param to_node: to node [x,y]
        :param max_dist: maximum distance between two nodes
        :return: new node [x,y]
        """
        dist = np.linalg.norm(to_node - from_node)
        if dist < max_dist:
            return to_node
        else:
            new = from_node + (to_node - from_node) * max_dist / dist
            new = np.round(new).astype(int)
            return new

    def _is_collision_free(self, from_node: np.ndarray, to_node: np.ndarray) -> bool:
        """
        Function to check if the path between two nodes is collision free.
        :param from_node: from node [x,y]
        :param to_node: to node [x,y]
        :return: True if collision free, False otherwise
        """
        if to_node is None or np.all(from_node == to_node):
            return True
        # get all points between from_node and to_node
        points_between = self._get_points_between(from_node, to_node)
        for point in points_between:
            if self.occupancy_grid[point[0], point[1]] == 1:
                return False
        return True

    def _get_points_between(self, src: np.ndarray, dst: np.ndarray) -> List[np.ndarray]:
        """
        Function to get all points between two nodes.
        :param from_node: from node [x,y]
        :param to_node: to node [x,y]
        :return: points between from_node and to_node
        """

        dx, dy = dst[0] - src[0], dst[1] - src[1]
        yaw = math.atan2(dy, dx)
        d = math.hypot(dx, dy)
        steps = np.arange(0, d, 0.2).reshape(-1, 1)
        pts = src + steps * np.array([math.cos(yaw), math.sin(yaw)])
        pts = np.vstack((pts, dst))

        # make a union of points ceiled and floored
        pts = np.unique(np.concatenate((np.ceil(pts), np.floor(pts))), axis=0).astype(int)
        # convert to list
        pts = pts.tolist()

        return pts

    def _find_near_nodes(self, node: np.ndarray, max_dist: float) -> List[np.ndarray]:
        """
        Function to find all nodes within a given distance of the given node.
        :param node: node [x,y]
        :param max_dist: maximum distance between two nodes
        :return: nodes within max_dist of the given node
        """
        # convert nodes to a KDTree for fast nearest neighbor search
        node_list = list(self.graph.nodes)
        node_kd_tree = cKDTree(node_list)
        # find nearest node
        near_nodes_idxs = node_kd_tree.query_ball_point(node, max_dist)
        near_nodes = [node_list[i] for i in near_nodes_idxs]
        # remove the node itself from near nodes
        near_nodes = [n for n in near_nodes if not np.all(n == node)]
        return near_nodes

    def _display_graph(
        self,
        graph: nx.Graph,
        sample=None,
        current_node=None,
        near_nodes=None,
        path1: List[Tuple] | None = None,
        path2: List[Tuple] | None = None,
        path3: List[Tuple] | None = None
    ) -> None:
        """
        Function to display the graph on the map.
        :param graph: graph
        :param start: start position [x,y]
        :param goal: goal position [x,y]
        """

        map = (self.occupancy_grid * 255).astype(np.float32)
        map_rgb = cv2.cvtColor(map, cv2.COLOR_GRAY2RGB)

        # increase size of image for better visualization
        map_rgb = cv2.resize(
            map_rgb,
            (
                map_rgb.shape[1] * DISPLAY_RESIZE_FACTOR,
                map_rgb.shape[0] * DISPLAY_RESIZE_FACTOR,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        # draw graph
        for edge in graph.edges:
            pt1 = (
                edge[0][1] * DISPLAY_RESIZE_FACTOR,
                edge[0][0] * DISPLAY_RESIZE_FACTOR,
            )
            pt2 = (
                edge[1][1] * DISPLAY_RESIZE_FACTOR,
                edge[1][0] * DISPLAY_RESIZE_FACTOR,
            )

            cv2.line(map_rgb, pt1, pt2, RED, 2)

        # draw start and goal
        if self.start is not None and self.goal is not None:
            cv2.circle(
                map_rgb,
                (self.start[1] * DISPLAY_RESIZE_FACTOR, self.start[0] * DISPLAY_RESIZE_FACTOR),
                3 * DISPLAY_RESIZE_FACTOR,
                BLUE,
                -1,
            )
            cv2.circle(
                map_rgb,
                (self.goal[1] * DISPLAY_RESIZE_FACTOR, self.goal[0] * DISPLAY_RESIZE_FACTOR),
                3 * DISPLAY_RESIZE_FACTOR,
                GREEN,
                -1,
            )

        # draw sample
        if sample is not None:
            cv2.circle(
                map_rgb,
                (sample[1] * DISPLAY_RESIZE_FACTOR, sample[0] * DISPLAY_RESIZE_FACTOR),
                1 * DISPLAY_RESIZE_FACTOR,
                (0, 255, 255),
                -1,
            )

        # draw current node
        if current_node is not None:
            cv2.circle(
                map_rgb,
                (
                    current_node[1] * DISPLAY_RESIZE_FACTOR,
                    current_node[0] * DISPLAY_RESIZE_FACTOR,
                ),
                1 * DISPLAY_RESIZE_FACTOR,
                (255, 0, 255),
                -1,
            )

        # highlight near nodes
        if near_nodes is not None:
            for near_node in near_nodes:
                cv2.circle(
                    map_rgb,
                    (
                        near_node[1] * DISPLAY_RESIZE_FACTOR,
                        near_node[0] * DISPLAY_RESIZE_FACTOR,
                    ),
                    1 * DISPLAY_RESIZE_FACTOR,
                    (255, 255, 0),
                    -1,
                )
        
        # draw path
        if path1 is not None:
            for i, node in enumerate(path1[:-1]):
                pt1 = (
                    node[1] * DISPLAY_RESIZE_FACTOR,
                    node[0] * DISPLAY_RESIZE_FACTOR,
                )
                pt2 = (
                    path1[i+1][1] * DISPLAY_RESIZE_FACTOR,
                    path1[i+1][0] * DISPLAY_RESIZE_FACTOR,
                )

                cv2.line(map_rgb, pt1, pt2, GREEN, 2)

        if path2 is not None:
            for i, node in enumerate(path2[:-1]):
                pt1 = (
                    node[1] * DISPLAY_RESIZE_FACTOR,
                    node[0] * DISPLAY_RESIZE_FACTOR,
                )
                pt2 = (
                    path2[i + 1][1] * DISPLAY_RESIZE_FACTOR,
                    path2[i + 1][0] * DISPLAY_RESIZE_FACTOR,
                )
                cv2.line(map_rgb, pt1, pt2, BLUE, 2)

        if path3 is not None:
            for i, node in enumerate(path3[:-1]):
                pt1 = (
                    node[1] * DISPLAY_RESIZE_FACTOR,
                    node[0] * DISPLAY_RESIZE_FACTOR,
                )
                pt2 = (
                    path3[i + 1][1] * DISPLAY_RESIZE_FACTOR,
                    path3[i + 1][0] * DISPLAY_RESIZE_FACTOR,
                )

                cv2.line(map_rgb, pt1, pt2, (255,255,255), 2)



        cv2.imshow("map", map_rgb)

    def _cost(self, node: np.ndarray) -> float:
        """
        Function to calculate the cost of a node.
        :param node1: node [x,y]
        :return: cost of the node
        """
        # check if node is in graph
        if tuple(node) not in self.graph.nodes:
            return np.inf
        return nx.shortest_path_length(self.graph, tuple(self.start), tuple(node), weight="weight")

    
    def _optimize_path(self, path: List[Tuple]) -> List[Tuple]:
        """
        Function to optimize the path.
        :param path: path
        :return: optimized path
        """
        if path is None or len(path) == 0:
            return path
        optimized_path = []
        start_node = path[0]

        optimized_path.append(start_node)
        for i, node in enumerate(path[1:]):
            # check if path between start_node and node is collision free
            if self._is_collision_free(np.array(start_node), np.array(node)):
                # remove all nodes between start_node and node
                continue
            else:
               start_node = path[i]
               optimized_path.append(start_node)
        optimized_path.append(path[-1])
        return optimized_path

    def _optimize_path_v2(self, path: List[Tuple]) -> List[Tuple]:
        """
        Function to optimize the path.
        Small improvement: we walk over the path forward AND backward.
        :param path: path
        :return: optimized path
        """
        def _path_cost(subpath: List[Tuple]) -> float:
            """
            Function to calculate the cost of a node.
            :param node1: node [x,y]
            :param node2: node [x,y]
            :return: cost of the node
            """
            cost = 0
            for i, point in enumerate(subpath[:-1]):
                cost += np.linalg.norm(np.array(point) - np.array(subpath[i+1])) # type: ignore
            return cost # type: ignore
        forward_path = self._optimize_path(path)
        backward_path = self._optimize_path(path[::-1])
        backward_path = backward_path[::-1]
        
        print(f"forward_path: {forward_path}")
        print(f"backward_path: {backward_path}")

        optimized_path = []

        # check intersection between forward and backward path
        intersection = set(forward_path[1:]).intersection(set(backward_path[1:]))
        intersection = list(intersection)
        intersection.sort(key=lambda x: forward_path.index(x))
        
        print(f"intersection: {intersection}")

        if len(intersection) == 0:
            backward_cost = _path_cost(backward_path)
            forward_cost = _path_cost(forward_path)
            return forward_path if forward_cost < backward_cost else backward_path

        prec_intersection_id_forward = 0
        prec_intersection_id_backward = 0
        for point in intersection:
            # get subpath before point
            subpath_forward = forward_path[prec_intersection_id_forward:forward_path.index(point)+1]
            subpath_backward = backward_path[prec_intersection_id_backward:backward_path.index(point)+1]

            # check cost of point in forward path
            cost_forward = _path_cost(subpath_forward)
            # check cost of point in backward path
            cost_backward = _path_cost(subpath_backward)
            
            print(f"subpath_forward: {subpath_forward}, subpath_backward: {subpath_backward}")
            print(f"cost forward: {cost_forward}, cost backward: {cost_backward}")


            if cost_forward < cost_backward:
                # remove all nodes between start_node and node
                optimized_path.extend(subpath_forward)
            else:
                optimized_path.extend(subpath_backward)
            prec_intersection_id_forward = forward_path.index(point)
            prec_intersection_id_backward = backward_path.index(point)
        return backward_path, forward_path, optimized_path

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        max_iter: int,
        max_dist: float,
        goal_dist_thresh: float = 20,
        stop_on_goal: bool = False,
    ) -> List[np.ndarray]:
        """
        Function to plan a path from start to goal on the given map.
        :param start: start position [x,y]
        :param goal: goal position [x,y]
        :param map: 2D occupancy grid map
        :param max_iter: maximum number of iterations
        :param max_dist: maximum distance between two nodes
        :return: path from start to goal
        """
        self.start = start
        self.goal = goal
        self.graph.add_node(tuple(start))
        goal_found = False
        rewire_count = 0

        while len(self.graph.nodes) < max_iter:
            sample = self._get_random_node()
            if self.occupancy_grid[sample[0], sample[1]] == 1:
                continue

            # find nearest node
            nearest_node = self._find_nearest_node(sample)
            new_node = self._steer(nearest_node, sample, max_dist)

            # check if new node is collision free
            if not self._is_collision_free(nearest_node, new_node):
                continue

            # add new node to graph
            self.graph.add_node(tuple(new_node))

            nb_nodes = len(self.graph.nodes)
            ball_radius = min(SCALING_FACTOR * np.sqrt(np.log(nb_nodes) / nb_nodes), max_dist)
            near_nodes = self._find_near_nodes(new_node, ball_radius)

            # Find the node in near_nodes that minimizes the cost of reaching it from the root node
            min_cost = self._cost(nearest_node) + np.linalg.norm(nearest_node - new_node)
            min_node = nearest_node
            for near_node in near_nodes:
                if not self._is_collision_free(near_node, new_node):
                    continue
                cost_new = self._cost(near_node) + np.linalg.norm(near_node - new_node)
                if cost_new < min_cost:
                    min_cost = cost_new
                    min_node = near_node

            # add edge to graph
            self.graph.add_edge(
                tuple(min_node),
                tuple(new_node),
                weight=np.linalg.norm(min_node - new_node),
            )
            # add parent to new node
            self.graph.nodes[tuple(new_node)]["parent"] = tuple(min_node)

            # rewire the graph
            for near_node in near_nodes:
                if not self._is_collision_free(near_node, new_node):
                    continue
                cost_new = self._cost(new_node)
                cost_near = self._cost(near_node)
                if cost_new + np.linalg.norm(near_node - new_node) < cost_near:
                    parr = self.graph.nodes[near_node]["parent"]
                    self.graph.remove_edge(parr, tuple(near_node))
                    self.graph.add_edge(
                        tuple(new_node),
                        tuple(near_node),
                        weight=np.linalg.norm(near_node - new_node),
                    )
                    self.graph.nodes[near_node]["parent"] = tuple(new_node)
                    rewire_count += 1

            if not goal_found:
                ### check if goal is reached
                # look for the nearest node to the goal
                nearest_node = self._find_nearest_node(goal)
                # check if the distance to the goal is less than the goal distance threshold
                if np.linalg.norm(nearest_node - goal) < goal_dist_thresh:
                    # add goal to the graph
                    self.graph.add_node(tuple(goal))
                    # add edge to graph
                    self.graph.add_edge(
                        tuple(nearest_node),
                        tuple(goal),
                        weight=np.linalg.norm(nearest_node - goal),
                    )
                    self.graph.nodes[tuple(goal)]["parent"] = tuple(nearest_node)
                    print("Path to goal reached!")
                    goal_found = True
                    if stop_on_goal:
                        break
        # find path
        print("rewire count: ", rewire_count)
        if tuple(goal) in self.graph.nodes:
            path = nx.shortest_path(self.graph, tuple(start), tuple(goal))
            # print(f"Unoptimized path: {path}")
            path1, path2, opti = self._optimize_path_v2(path)
            print(f"Optimized path: {path}")
            return path1, path2, opti, self.graph
        
        else:
            return [], self.graph


def main():
    # choose random seed
    np.random.seed(0)

    # load map
    map = cv2.imread("map_4.jpeg", cv2.IMREAD_GRAYSCALE)
    map = map / 255.0
    # convert map to np array
    map = np.array(map)
    # ceil map with threshold
    map[map > OCCUPANCY_THRESH] = 1
    map[map <= OCCUPANCY_THRESH] = 0
    map = map.astype(np.uint8)

    start = np.array([5, 5])
    goal = np.array([500, 26])

    # start = np.array([50, 50])
    # goal = np.array([99, 99])
    # map = np.zeros((100, 100))

    rrtstar_planner = RRTStar(map)
    start_time = time.process_time()
    path1, path2, opti, graph = rrtstar_planner.plan(start, goal, 3_000, 50, stop_on_goal=False, goal_dist_thresh=50)
    end_time = time.process_time()

    print("Time taken: ", end_time - start_time)

    # display path
    cv2.namedWindow("map", cv2.WINDOW_NORMAL)
    rrtstar_planner._display_graph(graph, start, goal, path1=path1, path2=path2, path3=opti)

    # cv2.imshow("map", map_rgb)
    while cv2.getWindowProperty("map", cv2.WND_PROP_VISIBLE) > 0:
        key = cv2.waitKey(100)
        if key != -1:
            break
    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
