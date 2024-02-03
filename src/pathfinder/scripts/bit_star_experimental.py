"""
summary:
Implementation of the BIT* algorithm for path planning in a 2D grid world.
"""

import numpy as np
import cv2
import math
import time
import pathlib
from typing import List, Set, Iterator
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Circle

BEIGE = (127, 173, 200)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)


class Vertex:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

    def point(self):
        return (self.x, self.y)


class BITStar:
    def __init__(
        self,
        occupancy_grid: np.ndarray,
        start: tuple,
        goal: tuple,
        max_batch: int = 10,
        batch_size: int = 100,
        stop_on_goal: bool = True,
        eta: float = 2,
        limit_secs=np.inf,
    ) -> None:
        self.occupancy_grid = occupancy_grid
        self.start = Vertex(start[0], start[1])
        self.goal = Vertex(goal[0], goal[1])
        self.max_batch = max_batch
        self.batch_size = batch_size
        self.stop_on_goal = stop_on_goal
        self.eta = eta
        self.limit_secs = limit_secs
        self.best_path = []
        self.curr_batch = 0
        self.curr_best_cost = np.inf
        self.samples, self.e_queue, self.v_queue, self.old_verts = (
            set(),
            set(),
            set(),
            set(),
        )
        self.V, self.E = set(), set()

    def plan(
        self,
        print_stats: bool = False,
    ) -> List[tuple]:
        # Timing variables
        self.time_prune = 0
        self.time_sample = 0
        self.time_expand = 0
        self.time_get_near = 0
        self.time_collision_check = 0
        self.time_eval_edge = 0
        self.time_g = 0

        # Init algo sets and graph
        self.samples, self.e_queue, self.v_queue, self.old_verts = (
            set(),
            set(),
            set(),
            set(),
        )
        self.V, self.E = set(), set()
        self.samples.add(self.goal)
        self.V.add(self.start)
        self.radius = np.inf
        self.g = {self.start: 0.0, self.goal: np.inf}

        batch_start_time = global_start_time = time.process_time()

        # save map, start and goal to image file in Desktop
        map_img = self.occupancy_grid.copy() * 255
        # convert to RGB
        map_img = cv2.cvtColor(map_img, cv2.COLOR_GRAY2RGB)
        # draw start and goal
        map_img[self.start.x, self.start.y] = RED
        map_img[self.goal.x, self.goal.y] = GREEN
        # save image
        path = str(pathlib.Path.home()) + "/Desktop/"
        cv2.imwrite(path + "map.png", map_img)

        # if start or goal are in obstacle, return empty path
        if self.occupancy_grid[self.start.x, self.start.y] == 1:
            print(f"Start in obstacle at {self.start.x, self.start.y}")
            return []
        if self.occupancy_grid[self.goal.x, self.goal.y] == 1:
            print(f"Goal in obstacle at {self.goal.x, self.goal.y}")
            return []

        # if goal is in line of sight of start, return path
        if self._is_collision_free(self.start, self.goal):
            self.goal.parent = self.start
            path = self._compute_path()
            # update best path and cost
            self.best_path = path
            self.curr_best_cost = self._g(self.goal)
            return path

        is_goal_reached = False
        while self.curr_batch < self.max_batch:
            
            yield self._compute_path()
            
            time_since_start = time.process_time() - global_start_time

            # Stop if timeout and path found
            if self._g(self.goal) < np.inf:
                if not is_goal_reached:
                    is_goal_reached = True
                    # print(f"Goal reached after {time_since_start:3.1f}s!!")
                if self.stop_on_goal:
                    return self._compute_path()

            # ==> NEW BATCH
            if len(self.e_queue) == 0 and len(self.v_queue) == 0:
                # update best path and cost
                self.best_path = self._compute_path()
                self.curr_best_cost = self._g(self.goal)

                if time_since_start > self.limit_secs:
                    print(f"Timeout after {time_since_start:3.1f}s!!")
                    return self._compute_path()

                if self.curr_batch > 0:
                    print(
                        f"Batch {self.curr_batch} -- best_cost: {self.curr_best_cost:3.1f}"
                    )
                    if print_stats:
                        self._print_and_reset_times(batch_start_time)

                self._prune(self._g(self.goal))
                nb_samples = (
                    self.batch_size * 3 if self.curr_batch == 0 else self.batch_size
                )
                new_samples = self._sample(nb_samples, self._g(self.goal))
                print(f"New samples: {len(new_samples)}")
                self.samples = self.samples | new_samples

                # label previous vertices as old
                self.old_verts = set(self.V)

                # requeue previous vertices for expansion
                self.v_queue = set(self.V)

                # update rewire radius based on new number of vertices
                self._update_radius(self._g(self.goal))

                self.curr_batch += 1
                batch_start_time = time.process_time()

            # while there are vertices with a cost-to-come lower than the best cost-to-goal in the edge queue
            while self._best_v_queue_value() <= self._best_e_queue_value():
                # get vertex with lowest cost-to-come
                self._expand_vertex(self._best_in_v_queue())

            # get best candidate edge
            (v_m, x_m) = self._best_in_e_queue()
            self.e_queue.remove((v_m, x_m))

            # * Evaluate edge
            time_start = time.process_time()
            # The edge (vm, xm), is first checked to see if it can improve the current solution.
            # If it cannot, then no other edges in the queue can so both queues are cleared to start a new batch
            cost_estim_1 = self._g(v_m) + self._c_hat(v_m, x_m) + self._h_hat(x_m)
            if not cost_estim_1 < self._g(self.goal):
                self.e_queue = set()
                self.v_queue = set()
                self.time_eval_edge += time.process_time() - time_start
                continue

            # Check if the edge can potentially improve the current solution, if not,
            actual_cost = self._c(v_m, x_m)
            cost_estim_2 = self._g_hat(v_m) + actual_cost + self._h_hat(x_m)
            if not cost_estim_2 < self._g(self.goal):
                self.time_eval_edge += time.process_time() - time_start
                continue

            # * Finally, we check if the edge improves the cost-to-come of its target vertex
            if self._g(v_m) + actual_cost < self._g(x_m):
                if x_m in self.V:
                    # x_m is in the tree, so it is a rewiring
                    # so we remove any edges to x_m in the graph
                    filter(lambda v, x: x == x_m, self.E)
                else:
                    # x_m is not in the tree, so it is an expansion
                    self.samples.remove(x_m)
                    self.V.add(x_m)
                    self.v_queue.add(x_m)

                # add edge to graph
                self.E.add((v_m, x_m))
                self.g[x_m] = self.g[v_m] + actual_cost
                x_m.parent = v_m

                # * prune edge_queue to remove edges that cannot improve cost-to-come of x_m
                # remove edges in edge_queue that cannot improve cost-to-come of x_m
                for v, x in list(self.e_queue):
                    # ignore edges that do not involve x_m
                    if x_m != x:
                        continue
                    if self._g(v) + self._c_hat(v, x_m) >= self._g(x_m):
                        self.e_queue.remove((v, x))
            self.time_eval_edge += time.process_time() - time_start

        return self._compute_path()

    def _compute_path(self):
        path = []
        if self.goal.parent is not None:
            curr_node = self.goal
            path.append((curr_node.x, curr_node.y))
            while curr_node.parent is not None:
                curr_node = curr_node.parent
                path.append((curr_node.x, curr_node.y))
        path.reverse()
        return path

    def _expand_vertex(self, v: Vertex) -> None:
        """Expands the given vertex by adding edges to the graph

        Args:
            vertex (tuple): vertex to expand
        """
        time_start = time.process_time()

        self.v_queue.remove(v)
        # Edges to unconnected states are always added to edge queue if they could be part of a better solution
        for x in self._get_near_points(v, self.samples):
            g_hat_v = self._g_hat(v)
            c_hat_vx = self._c_hat(v, x)
            h_hat_x = self._h_hat(x)
            if g_hat_v + c_hat_vx + h_hat_x < self._g(self.goal):
                self.g[x] = np.inf
                self.e_queue.add((v, x))

        if v not in self.old_verts:
            # Edges to connected states are added to edge queue if they could improve the cost-to-come of the state
            for w in self._get_near_points(v, self.V):
                g_hat_v = self._g_hat(v)
                c_hat_vw = self._c_hat(v, w)
                h_hat_w = self._h_hat(w)
                if (
                    (v, w) not in self.E
                    and g_hat_v + c_hat_vw + h_hat_w < self._g(self.goal)
                    and self._g(v) + c_hat_vw < self._g(w)
                ):
                    self.e_queue.add((v, w))
                    if w not in self.g:
                        self.g[w] = np.inf

        self.time_expand += time.process_time() - time_start

    def _prune(self, best_cost: float) -> None:
        """
        Removes nodes that cannot provide a solution better than the given cost
        Args:
            cost_thresh (float): cost threshold
        """
        time_start = time.process_time()

        # only keep samples that can provide a better solution
        self.samples = {
            sample for sample in self.samples if self._f_hat(sample) < best_cost
        }

        # only keep vertices that can provide a better solution
        self.V = {v for v in self.V if self._f_hat(v) <= best_cost}

        # only keep edges that can provide a better solution
        edges_to_delete = {
            (v, w)
            for (v, w) in self.E
            if self._f_hat(v) > best_cost or self._f_hat(w) > best_cost
        }
        self.E = self.E - edges_to_delete

        # find unconnected vertices (do not take into account start and goal)
        # remove unconnected vertices from graph
        unconnected_vertices = {v for v in self.V if self._g(v) == np.inf}
        self.V = self.V - unconnected_vertices
        # recycle them as samples
        self.samples = self.samples | unconnected_vertices

        self.time_prune += time.process_time() - time_start

    def _is_collision_free(self, from_node: Vertex, to_node: Vertex) -> bool:
        # retrieve cells in line between from_node and to_node
        # check if any cell is occupied
        time_start = time.process_time()
        line_gen = bresenham(from_node.point(), to_node.point())
        for cell in line_gen:
            if self.occupancy_grid[cell] == 1:
                self.time_collision_check += time.process_time() - time_start
                return False
        self.time_collision_check += time.process_time() - time_start
        return True

    def _sample(self, nb_samples: int, best_cost_to_goal: float) -> Set[Vertex]:
        """Returns a list of samples inside the ellipse whose focus are the start and goal

        Args:
            cost_to_goal (float): cost to goal of the best path found so far
        """
        time_start = time.process_time()
        if best_cost_to_goal == float("inf"):
            self.time_sample += time.process_time() - time_start
            return self._sample_uniform(self.batch_size)

        # Generate samples with rejection sampling
        samples = set()
        timeout = time.time() + 0.1
        while len(samples) < nb_samples:
            sample_coord = tuple(np.random.randint((0, 0), self.occupancy_grid.shape))
            sample = Vertex(sample_coord[0], sample_coord[1])
            # check if sample is inside ellipse
            if self._f_hat(sample) > best_cost_to_goal:
                continue
            if self.occupancy_grid[sample_coord] == 0:
                samples.add(sample)
            # timeout
            # if time.time() > timeout:
            #     return set()

        self.time_sample += time.process_time() - time_start
        return samples

    def _sample_uniform(self, num_samples: int) -> Set[Vertex]:
        samples = set()
        while len(samples) < num_samples:
            sample_coord = tuple(np.random.randint((0, 0), self.occupancy_grid.shape))
            if self.occupancy_grid[sample_coord] == 0:
                new_sample = Vertex(sample_coord[0], sample_coord[1])
                samples.add(new_sample)
        return samples

    def _get_near_points(
        self, vertex: Vertex, search_set: List[Vertex]
    ) -> List[Vertex]:
        start_time = time.process_time()
        near_points = [v for v in search_set if self._c_hat(vertex, v) <= self.radius]
        self.time_get_near += time.process_time() - start_time
        return near_points

    def _best_in_v_queue(self):
        if not self.v_queue:
            print("v_queue is empty")
            return None
        v_value = {v: self._g(v) + self._h_hat(v) for v in self.v_queue}
        return min(v_value, key=v_value.get)

    def _update_radius(self, best_c_to_goal: float) -> None:
        vertex_count = len(self.V) + len(self.samples)

        # Area of space that could provide a solution better than the current best (ellipse area)
        space_area = self.occupancy_grid.shape[0] * self.occupancy_grid.shape[1]
        if best_c_to_goal < np.inf:
            c = math.dist(self.start.point(), self.goal.point()) / 2
            a = best_c_to_goal / 2
            b = math.sqrt(a**2 - c**2)
            space_area = math.pi * a * b
        unit_disk_area = math.pi * 1**2

        self.radius = self.eta * np.sqrt(
            3 * space_area / unit_disk_area * np.log(vertex_count) / vertex_count
        )

    def _best_e_queue_value(self) -> float:
        if len(self.e_queue) == 0:
            return np.inf

        min_value = min(
            [self._g(v) + self._c_hat(v, x) + self._h_hat(x) for (v, x) in self.e_queue]
        )
        return min_value

    def _best_v_queue_value(self):
        if len(self.v_queue) == 0:
            return np.inf
        min_value = min([self._g(v) + self._h_hat(v) for v in self.v_queue])
        return min_value

    def _best_in_e_queue(self):
        if not self.e_queue:
            print("e_queue is empty")
            return None

        e_value = {
            (v, x): self._g(v) + self._c_hat(v, x) + self._h_hat(x)
            for v, x in self.e_queue
        }
        return min(e_value, key=e_value.get)

    def _g(self, vertex: Vertex) -> float:
        """Returns the cost-to-come of the given vertex

        Args:
            vertex (tuple): vertex to compute cost-to-come for
        """
        return self.g[vertex]

    def _g_hat(self, vertex: Vertex) -> float:
        """Returns the estimated cost-to-come of the given vertex

        Args:
            vertex (tuple): vertex to compute estimated cost-to-come for
        """
        return self._c_hat(self.start, vertex)

    def _h_hat(self, vertex: Vertex) -> float:
        """Returns the estimated cost-to-goal of the given vertex

        Args:
            vertex (tuple): vertex to compute estimated cost-to-goal for
        """
        return self._c_hat(vertex, self.goal)

    def _f_hat(self, vertex: Vertex) -> float:
        """Returns the estimated cost of the given vertex

        Args:
            vertex (tuple): vertex to compute estimated cost for
        """
        return self._g_hat(vertex) + self._h_hat(vertex)

    def _c(self, vertex1: Vertex, vertex2: Vertex) -> float:
        """Returns the cost of the edge between the given vertices

        Args:
            vertex1 (tuple): first vertex
            vertex2 (tuple): second vertex
        """
        if self._is_collision_free(vertex1, vertex2):
            return self._c_hat(vertex1, vertex2)
        else:
            return np.inf

    def _c_hat(self, vertex1: Vertex, vertex2: Vertex) -> float:
        """Returns the estimated cost of the edge between the given vertices

        Args:
            vertex1 (tuple): first vertex
            vertex2 (tuple): second vertex
        """
        pt1 = (vertex1.x, vertex1.y)
        pt2 = (vertex2.x, vertex2.y)
        return math.dist(pt1, pt2)

    def _print_and_reset_times(self, batch_start_time):
        # print timings in % of total time in batch
        batch_total = time.process_time() - batch_start_time
        print(f"\tPrune: {self.time_prune / batch_total * 100:3.1f}%")
        print(f"\tSample: {self.time_sample / batch_total * 100:3.1f}%")
        print(f"\tExpand: {self.time_expand / batch_total * 100:3.1f}%")
        print(f"\tGet near: {self.time_get_near / batch_total * 100:3.1f}%")
        print(
            f"\tCollision check: {self.time_collision_check / batch_total * 100:3.1f}%"
        )
        print(f"\tEval edge: {self.time_eval_edge / batch_total * 100:3.1f}%")
        print(f"\tg_cost: {self.time_g / batch_total * 100:3.1f}%")
        print(f"\tRadius: {self.radius:3.1f}")
        print(f"\tTotal: {batch_total:3.1f}s")

        # reset timings
        self.time_prune = 0
        self.time_collision_check = 0
        self.time_expand = 0
        self.time_get_near = 0
        self.time_sample = 0
        self.time_eval_edge = 0
        self.time_g = 0


def bresenham(p1, p2):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).

    Input coordinates should be integers.

    The result will contain both the start and the end point.
    """
    (x0, y0), (x1, y1) = p1, p2
    dx, dy = (x1 - x0), (y1 - y0)

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx, dy = abs(dx), abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2 * dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x * xx + y * yx, y0 + x * xy + y * yy
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy


###################################################################################################
###################################################################################################
###################################################################################################

class BitStarVisualizer:
    def __init__(self, bit_star_planner: BITStar):
        self.bit_star_planner = bit_star_planner
        self.fig, self.ax = plt.subplots()

    def init_plt(self):
        # set fig size according to map size
        self.fig.set_size_inches(
            self.bit_star_planner.occupancy_grid.shape[0] / 30,
            self.bit_star_planner.occupancy_grid.shape[1] / 30,
        )
        self.ax.set_xlim(0, self.bit_star_planner.occupancy_grid.shape[0])
        self.ax.set_ylim(0, self.bit_star_planner.occupancy_grid.shape[1])
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        # increase label size
        self.ax.xaxis.label.set_size(16)
        self.ax.yaxis.label.set_size(16)
        # self.ax.set_aspect("equal")

        # Transform occupancy grid to fit matplotlib imshow
        grid = self.bit_star_planner.occupancy_grid.copy().T

        # Draw map
        self.ax.imshow(grid, cmap="binary")

        # Draw start and goal
        start = (self.bit_star_planner.start.x, self.bit_star_planner.start.y)
        goal = (self.bit_star_planner.goal.x, self.bit_star_planner.goal.y)
        self.ax.add_patch(Circle(start, 5, color="r", fill=True, zorder=5))
        self.ax.add_patch(Circle(goal, 5, color="g", fill=True, zorder=5))

        # init collections
        self.edge_collec = LineCollection([], color="grey", linewidth=1, zorder=1)
        self.ax.add_collection(self.edge_collec)
        self.vertex_collec = PatchCollection([], color="orange", zorder=3)
        self.ax.add_collection(self.vertex_collec)
        self.sample_collec = PatchCollection([], color="b", zorder=2)
        self.ax.add_collection(self.sample_collec)
        self.path_lines = LineCollection([], color="g", linewidth=2, zorder=5)
        self.ax.add_collection(self.path_lines)

        # Anontation beneath the plot to show batch number
        self.anot_nb_batch = self.ax.annotate(
            f"Batch: {self.bit_star_planner.curr_batch}",
            xy=(0.5, -0.3),
            xycoords="axes fraction",
            fontsize=16,
            va="center",
            ha="center",
        )
        # Title
        self.ax.set_title("BIT*")

        # pack layout
        self.fig.tight_layout()
        return (
            self.edge_collec,
            self.vertex_collec,
            self.sample_collec,
            self.path_lines,
            self.anot_nb_batch,
        )

    def update_plt(self, path):
        # update collections with data from bit_star_planner
        self.edge_collec.set_segments(
            [
                [(edge[0].x, edge[0].y), (edge[1].x, edge[1].y)]
                for edge in self.bit_star_planner.E
            ]
        )
        self.vertex_collec.set_paths(
            [Circle((vertex.x, vertex.y), 1) for vertex in self.bit_star_planner.V]
        )
        self.sample_collec.set_paths(
            [
                Circle((sample.x, sample.y), 1)
                for sample in self.bit_star_planner.samples
            ]
        )

        path_segments = []
        for i in range(len(self.bit_star_planner.best_path) - 1):
            p1 = self.bit_star_planner.best_path[i]
            p2 = self.bit_star_planner.best_path[i + 1]
            path_segments.append([(p1[0], p1[1]), (p2[0], p2[1])])

        self.path_lines.set_segments(path_segments)
        self.anot_nb_batch.set_text(
            f"Batch: {self.bit_star_planner.curr_batch} -- best_cost: {self.bit_star_planner.curr_best_cost:3.1f}"
        )

        plt_objs = (
            self.edge_collec,
            self.vertex_collec,
            self.sample_collec,
            self.path_lines,
            self.anot_nb_batch,
        )
        return plt_objs
    
###################################################################################################
###################################################################################################
###################################################################################################

def main():
    
    OCCUPANCY_THRESH = 0.8

    # np.random.seed(0)

    # open map from png file on desktop
    path = str(pathlib.Path.home()) + "/Desktop/"
    map = cv2.imread(path + "map.png", cv2.IMREAD_COLOR)
    print(f"Orig Map shape: {map.shape}")
    
    # retrieve start and goal from map (red and green pixels)
    start_find = np.where(np.all(map == RED, axis=-1))
    start = (start_find[0][0], start_find[1][0])
    print(f"Start pixel: {start}")
    goal_find = np.where(np.all(map == GREEN, axis=-1))
    goal = (goal_find[0][0], goal_find[1][0])
    print(f"Goal pixel: {goal}")
    
    # convert to greyscale
    map = cv2.cvtColor(map, cv2.COLOR_RGBA2GRAY)
    print(f"Greyscale Map shape: {map.shape}")
    # replace start and goal pixels with 0
    map[start] = 0
    map[goal] = 0
    # normalize map to [0,1]
    map = map / 255.0
    # map = np.array(map)
    
    # ceil map with threshold
    map[map > OCCUPANCY_THRESH] = 1
    map[map <= OCCUPANCY_THRESH] = 0
    map = map.astype(np.uint8)
    
    bit_star = BITStar(
        map, start, goal, batch_size=50, max_batch=30, stop_on_goal=False, limit_secs=60
    )
    time_start = time.process_time()

    path_gen = bit_star.plan()
    bit_star_vis = BitStarVisualizer(bit_star)

    anim = FuncAnimation(  # noqa: F841
        fig=bit_star_vis.fig,
        func=bit_star_vis.update_plt,
        init_func=bit_star_vis.init_plt,
        frames=path_gen,
        repeat=False,
        save_count=bit_star.max_batch,
    )
    plt.show()
    # save animation to mp4 file on desktop
    # folder = str(pathlib.Path.home()) + "/Desktop/"
    # anim.save(folder + 'bit_star.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    print(f"Total time: {time.process_time() - time_start:3.1f}s")




if __name__ == "__main__":
    main()
