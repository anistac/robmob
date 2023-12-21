from typing import List
import rospy
from astar import AStar
import numpy as np

class Node():
    _x: int # vertical
    _y: int # horizontal
    _occupancy: float

    def __init__(self, x:int, y:int, occupancy: float) -> None:
        self._x = x
        self._y = y
        self._occupancy = occupancy


class PathFinder(AStar):
    occupancy_grid: np.ndarray
    def __init__(self, occupancy_grid: np.ndarray):
        super(PathFinder, self).__init__()
        self.occupancy_grid = occupancy_grid 
        
    def neighbors(self, node: Node):
        neighbors: List[Node] = []
        if node._x + 1 < self.occupancy_grid.shape[0]:
            neighbors += self.occupancy_grid[node._x + 1, node._y] # right
        if node._x -1 >= 0:
            neighbors += self.occupancy_grid[node._x - 1, node._y] # left
        if node._y + 1 < self.occupancy_grid.shape[1]:
            neighbors += self.occupancy_grid[node._x, node._y + 1] # up
        if node._y -1 >= 0: 
            neighbors += self.occupancy_grid[node._x, node._y - 1] # down
        
        # filter neighbours with occupancy > 0.8
        thresh =.8
        neighbors = [n for n in neighbors if n._occupancy > thresh]

        return neighbors

    def distance_between(self, node1: Node, node2: Node):
        return  1

    def heuristic_cost_estimate(self, current: Node, goal: Node):
        current_pos = np.array([current._x, current._y])
        goal_pos = np.array([goal._x, goal._y])
        return np.linalg.norm(current_pos - goal_pos)


    def is_goal_reached(self, current, goal):
        return current == goal
        pass
