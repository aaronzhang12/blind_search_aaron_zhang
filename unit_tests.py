import unittest
from typing import Optional, Callable, Any, Tuple, List
from maze import Maze, MazeRoom
from search import bfs, dfs

class IOTest(unittest.TestCase):
    """
    Tests IO for bfs and dfs implementations. Contains basic/trivial test cases.

    Each test instatiates a Maze object and tests the bfs and dfs algorithms on it.
    Note: The tests currently are only testing if the algorithms start and end at the
    correct locations and if the path returned is the correct length (optional).

    You may wish to add path validity checks. How do you know if a path is valid nor not?
    Your path should not teleport you, move through a wall, or go through a cell more than once.
    Maybe you should test for this... We certainly will during grading!

    These tests are not exhaustive and do not check if your implementation follows the
    algorithm correctly. We encourage you to create your own tests as necessary.
    """

    def _check_maze(self, algorithm: Callable[[Maze], Tuple[Optional[List[Any]], dict]], maze: Maze, length: Optional[int] = None) -> None:
        """
        Test algorithm on a Maze
        algorithm: algorithm to test
        maze: Maze to test algorithm on
        length: length that the path returned from algorithm should be.
                Think about why this argument is optional, and when you should provide it.
        """
        path = algorithm(maze)[0]
        self.assertIsNotNone(path, "Algorithm should return a valid path")
        
        if path is not None:
            self.assertEqual(path[0], maze.get_start_state(),
                             "Path should start with the start state")
            self.assertTrue(maze.is_goal_state(path[-1]),
                            "Path should end with the goal state")
            if length:
                self.assertEqual(len(path), length,
                                 f"Path length should be {length}")
                
    def _assert_valid_path(self, maze: Maze, path: List[Any]) -> None:
        # 2) make sure that all next states ae neighbors of current states
        for u, v in zip(path, path[1:]):
            self.assertIn(v, maze.get_successors(u), f"Step {u}->{v} is not a legal move")

        # 3) Returns a simple path (no revisits)
        self.assertEqual(len(path), len(set(path)), "Path revisits a state")

    def _validate_stats(self, stats: dict, path: List[Any], width: int, height: int):
        area = width * height

        # path_length is number of edges (path length - 1)
        self.assertEqual(stats["path_length"], max(0, len(path) - 1),
                         "stats['path_length'] should be len(path) - 1")

        # expanded/frontier should be positive and bounded by area
        self.assertGreater(stats["states_expanded"], 0, "states_expanded should be > 0")
        self.assertGreater(stats["max_frontier_size"], 0, "max_frontier_size should be > 0")
        self.assertLessEqual(stats["states_expanded"], area, "expanded should not exceed maze area")
        self.assertLessEqual(stats["max_frontier_size"], area, "frontier should not exceed maze area")

        # expanded should be at least the nodes along the path (rough sanity)
        self.assertGreaterEqual(stats["states_expanded"], max(1, len(path) - 1))


    def test_bfs(self) -> None:
        single_cell_maze = Maze(1, 1)
        self._check_maze(bfs, single_cell_maze, 1)
        # check that the start and goal are equal
        self.assertEqual(single_cell_maze.start_state, single_cell_maze.goal_state)
        path, stats = bfs(single_cell_maze)
        self._assert_valid_path(single_cell_maze, path)
        self.assertEqual(len(path), 1)
        self.assertEqual(stats["path_length"], 0)
        self._validate_stats(stats, path, 1, 1)

        two_by_two_maze = Maze(2, 2)
        self._check_maze(bfs, two_by_two_maze, 3)

        large_maze = Maze(10, 10)
        self._check_maze(bfs, large_maze)
        
    def test_dfs(self) -> None:
        single_cell_maze = Maze(1, 1)
        self._check_maze(dfs, single_cell_maze, 1)

        two_by_two_maze = Maze(2, 2)
        self._check_maze(dfs, two_by_two_maze, 3)

        large_maze = Maze(10, 10)
        self._check_maze(dfs, large_maze)

    def test_no_empty_path_when_reachable(self):
        # Build a 1x3 corridor: (0,0) -> (0,1) -> (0,2)
        board = [[MazeRoom() for _ in range(3)] for _ in range(1)]
        # knock down east/west walls to open the corridor
        # (0,0) <-> (0,1)
        board[0][0].east = 0
        board[0][1].west = 0
        # (0,1) <-> (0,2)
        board[0][1].east = 0
        board[0][2].west = 0

        m = Maze(width=3, height=1, start=(0,0), goal=(0,2), self_generating=False, board=board)

        bpath, bstats = bfs(m)
        dpath, dstats = dfs(m)

        # Checking that an actual path is outputed with positive length
        self.assertIsNotNone(bpath, "BFS returned None on a reachable maze")
        self.assertIsNotNone(dpath, "DFS returned None on a reachable maze")
        self.assertGreater(len(bpath), 0, "BFS returned an empty path on a reachable maze")
        self.assertGreater(len(dpath), 0, "DFS returned an empty path on a reachable maze")

        # start and goal
        self.assertEqual(bpath[0], m.get_start_state())
        self.assertTrue(m.is_goal_state(bpath[-1]))
        self.assertEqual(dpath[0], m.get_start_state())
        self.assertTrue(m.is_goal_state(dpath[-1]))

        # exact length: nodes=3, edges=2
        self.assertEqual(len(bpath), 3)
        self.assertEqual(bstats["path_length"], 2)

if __name__ == "__main__":
    unittest.main()