from nav_msgs.msg import MapMetaData
from typing import Tuple

class GridWorldTransform():
    """
    This class provides methods to transform coordinates between the map and pixel space.
    """

    def update(self, map_metadata: MapMetaData):
        """
        Updates the map metadata.

        Args:
            map_metadata (MapMetaData): The map metadata containing resolution and origin information.
        """
        self.map_resolution = map_metadata.resolution
        self.map_x_origin = map_metadata.origin.position.x
        self.map_y_origin = map_metadata.origin.position.y
        
    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int]:
        """
        Converts world coordinates to pixel coordinates.

        Args:
            world_x (float): The x-coordinate in the world space.
            world_y (float): The y-coordinate in the world space.

        Returns:
            Tuple[int]: The pixel coordinates (px, py).
        """
        px = int((world_x - self.map_x_origin) / self.map_resolution)
        py = int((world_y - self.map_y_origin) / self.map_resolution)
        
        return (px, py)
   
    def grid_to_world(self, px: int, py: int) -> Tuple[float]:
        """
        Converts pixel coordinates to world coordinates.

        Args:
            px (int): The x-coordinate in the pixel space.
            py (int): The y-coordinate in the pixel space.

        Returns:
            Tuple[float]: The world coordinates (world_x, world_y).
        """
        world_x = px * self.map_resolution + self.map_x_origin
        world_y = py * self.map_resolution + self.map_y_origin

        return (world_x, world_y)
