import unittest
from typing import Optional, Callable, Any, Tuple, List, Set
from maze import Maze, MazeRoom
from search import bfs, dfs
import random
from directed_graph import DirectedGraph

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

        # path_length is number of nodes (path length)
        self.assertEqual(stats["path_length"], len(path),
                         "stats['path_length'] should be len(path)")

        # expanded/frontier should be positive and bounded by area
        self.assertGreater(stats["states_expanded"], 0, "states_expanded should be > 0")
        self.assertGreater(stats["max_frontier_size"], 0, "max_frontier_size should be > 0")
        self.assertLessEqual(stats["states_expanded"], area, "expanded should not exceed maze area")
        self.assertLessEqual(stats["max_frontier_size"], area, "frontier should not exceed maze area")

        # expanded should be at least the nodes along the path (rough sanity)
        self.assertGreaterEqual(stats["states_expanded"], len(path))

        # path length should be less than the area of the grid
        self.assertLessEqual(stats["path_length"], area)


    def test_bfs(self) -> None:
        random.seed(0)
        single_cell_maze = Maze(1, 1)
        self._check_maze(bfs, single_cell_maze, 1)
        # check that the start and goal are equal
        self.assertEqual(single_cell_maze.start_state, single_cell_maze.goal_state)
        path, stats = bfs(single_cell_maze)
        self._assert_valid_path(single_cell_maze, path)
        self.assertEqual(len(path), 1)
        self.assertEqual(stats["path_length"], 1)
        self._validate_stats(stats, path, 1, 1)

        random.seed(0)
        two_by_two_maze = Maze(2, 2)
        self._check_maze(bfs, two_by_two_maze, 3)
        path, stats = bfs(two_by_two_maze)
        self._assert_valid_path(two_by_two_maze, path)
        # path should be exactly 3
        self.assertEqual(len(path), 3)
        self._validate_stats(stats, path, 2, 2)

        random.seed(0)
        large_maze = Maze(10, 10)
        self._check_maze(bfs, large_maze)
        path, stats = bfs(large_maze)
        self._assert_valid_path(large_maze, path)
        # path should be at least 19
        self.assertGreaterEqual(len(path), 19)
        self._validate_stats(stats, path, 10, 10)
        
    def test_dfs(self) -> None:
        random.seed(0)
        single_cell_maze = Maze(1, 1)
        self._check_maze(dfs, single_cell_maze, 1)

        # path and stats should be same/similar to bfs for simple maze
        self.assertEqual(single_cell_maze.start_state, single_cell_maze.goal_state)
        path, stats = dfs(single_cell_maze)
        self._assert_valid_path(single_cell_maze, path)
        self.assertEqual(len(path), 1)
        self.assertEqual(stats["path_length"], 1)
        self._validate_stats(stats, path, 1, 1)

        random.seed(0)
        two_by_two_maze = Maze(2, 2)
        self._check_maze(dfs, two_by_two_maze, 3)

        # path and stats should be similar to bfs for 2x2 maze
        path, stats = dfs(two_by_two_maze)
        self._assert_valid_path(two_by_two_maze, path)
        # path should be exactly 3
        self.assertEqual(len(path), 3)
        self._validate_stats(stats, path, 2, 2)

        # path should be similar to bfs
        random.seed(0)
        large_maze = Maze(10, 10)
        self._check_maze(dfs, large_maze)
        path, stats = dfs(large_maze)
        self._assert_valid_path(large_maze, path)
        # path should be at least 19
        self.assertGreaterEqual(len(path), 19)
        self._validate_stats(stats, path, 10, 10)

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
        self.assertEqual(bstats["path_length"], 3)

    def test_unreachable_returns_none(self):
        # 2x2 board with all walls intact; start != goal; self_generating=False
        board = [[MazeRoom() for _ in range(2)] for _ in range(2)]
        m = Maze(width=2, height=2, start=(0,0), goal=(1,1),
                self_generating=False, board=board)

        bpath, bstats = bfs(m)
        dpath, dstats = dfs(m)

        self.assertIsNone(bpath, "BFS should return None when goal is unreachable")
        self.assertIsNone(dpath, "DFS should return None when goal is unreachable")

        # Reasonable stats: expanded at least the start, frontier tracked
        self.assertGreaterEqual(bstats["states_expanded"], 1)
        self.assertGreaterEqual(dstats["states_expanded"], 1)
        self.assertGreaterEqual(bstats["max_frontier_size"], 1)
        self.assertGreaterEqual(dstats["max_frontier_size"], 1)


    def test_bfs_not_longer_than_dfs(self):
        random.seed(0)
        m = Maze(10, 10)
        bpath, _ = bfs(m)
        dpath, _ = dfs(m)
        self.assertIsNotNone(bpath)
        self.assertIsNotNone(dpath)
        self.assertLessEqual(len(bpath), len(dpath),
                            "BFS path (nodes) should be <= DFS path")
        
    def test_bfs_shortest_on_5x5_many_solutions(self):
        # Fully open 5x5 grid: every adjacent pair linked both ways
        W = H = 5
        board = [[MazeRoom() for _ in range(W)] for _ in range(H)]

        def link(r1, c1, r2, c2, d1, d2):
            setattr(board[r1][c1], d1, 0)
            setattr(board[r2][c2], d2, 0)

        # Open all horizontal edges
        for r in range(H):
            for c in range(W - 1):
                link(r, c, r, c + 1, "east", "west")

        # Open all vertical edges
        for r in range(H - 1):
            for c in range(W):
                link(r, c, r + 1, c, "south", "north")

        m = Maze(width=W, height=H, start=(0, 0), goal=(H - 1, W - 1),
                self_generating=False, board=board)

        bpath, bstats = bfs(m)
        dpath, dstats = dfs(m)

        # must find paths
        self.assertIsNotNone(bpath)
        self.assertIsNotNone(dpath)

        # BFS must be shortest: nodes = 9 (edges 8)
        self.assertEqual(len(bpath), 9, "BFS did not return a shortest path in an open 5x5 grid")
        self.assertEqual(bstats["path_length"], 9, "BFS path_length should equal number of nodes")

        # DFS can be >= 9; don’t force it to be longer, but BFS must be <= DFS
        self.assertLessEqual(len(bpath), len(dpath), "BFS path should be <= DFS path length")

        # legality + in-bounds for BFS
        for u, v in zip(bpath, bpath[1:]):
            self.assertIn(v, m.get_successors(u), f"BFS illegal step {u}->{v}")
        self.assertEqual(len(bpath), len(set(bpath)), "BFS path revisits a state")
        for s in bpath:
            r, c = s.location
            self.assertTrue(0 <= r < m.height and 0 <= c < m.width,
                            f"BFS out-of-bounds state {s.location}")


    def test_path_never_out_of_bounds(self):
        random.seed(0)  # stabilize the generated maze
        m = Maze(10, 10)

        bpath, _ = bfs(m)
        self.assertIsNotNone(bpath, "BFS should find a path in this maze")
        for s in bpath:
            r, c = s.location
            self.assertTrue(0 <= r < m.height and 0 <= c < m.width,
                            f"BFS produced out-of-bounds state {s.location}")

        dpath, _ = dfs(m)
        self.assertIsNotNone(dpath, "DFS should find a path in this maze")
        for s in dpath:
            r, c = s.location
            self.assertTrue(0 <= r < m.height and 0 <= c < m.width,
                            f"DFS produced out-of-bounds state {s.location}")
            
    def _dg(self) -> DirectedGraph:
        """
        Build a 10x10 directed graph with:
          - Short path: 0 -> 1 -> 3
          - Long path:  0 -> 9 -> 8 -> 7 -> 6 -> 5 -> 4 -> 3
        All other entries are None (no edge).
        """
        N = 10
        none = None
        # start with all None
        M: List[List[Optional[float]]] = [[none for _ in range(N)] for _ in range(N)]

        # short branch edges
        M[0][1] = 1.0
        M[1][3] = 1.0

        # long branch edges
        M[0][9] = 1.0
        M[9][8] = 1.0
        M[8][7] = 1.0
        M[7][6] = 1.0
        M[6][5] = 1.0
        M[5][4] = 1.0
        M[4][3] = 1.0

        goals: Set[int] = {3}
        return DirectedGraph(M, goal_indices=goals, start_state=0)

    def _assert_path_in_bounds_and_legal(self, g: DirectedGraph, path):
        self.assertIsNotNone(path, "Algorithm returned None on a reachable graph")
        self.assertGreater(len(path), 0, "Algorithm returned empty path on a reachable graph")
        # start and goal
        self.assertEqual(path[0], g.get_start_state(), "Path must start at the start state")
        self.assertTrue(g.is_goal_state(path[-1]), "Path must end at a goal state")

        # in-bounds nodes and legal directed edges
        n = len(g.matrix)
        for u in path:
            self.assertTrue(0 <= u < n, f"Out-of-bounds state {u}")
        for u, v in zip(path, path[1:]):
            self.assertIsNotNone(g.matrix[u][v], f"Illegal step {u}->{v} (no directed edge)")

        # no revisits (simple path)
        self.assertEqual(len(path), len(set(path)), "Path revisits a state")

    def test_bfs_queue_vs_stack(self):
        """
        BFS must return the shortest path (nodes = 3: [0,1,3]).
        If BFS mistakenly uses a stack (pop instead of popleft), it will likely follow
        0->9->8->...->3 (8 nodes) and FAIL this test.
        """
        g = self._dg()
        path, stats = bfs(g)

        self._assert_path_in_bounds_and_legal(g, path)
        self.assertEqual(len(path), 3, "BFS did not return the shortest (3-node) path")
        # Your convention: path_length equals number of nodes
        self.assertEqual(stats["path_length"], 3, "BFS stats['path_length'] must equal len(path)")

    def test_dfs_stack_vs_queue(self):
        """
        On this graph, correct DFS (stack) will take the long detour (>= 8 nodes).
        If DFS mistakenly uses a queue (popleft), it will tend to match BFS's 3-node path, and FAIL.
        """
        g = self._dg()
        bpath, _ = bfs(g)
        dpath, dstats = dfs(g)

        self._assert_path_in_bounds_and_legal(g, dpath)

        # DFS should NOT be shorter than BFS here (and usually will be strictly longer)
        self.assertGreaterEqual(len(dpath), len(bpath),
                                "DFS path unexpectedly shorter than BFS — DFS may be using a queue")
        # In this specific graph with the given successor order, DFS should hit the long chain.
        self.assertGreaterEqual(len(dpath), 8,
                                "DFS should explore the long branch first on this graph (stack behavior)")

        self.assertEqual(dstats["path_length"], len(dpath),
                         "DFS stats['path_length'] must equal len(path)")

if __name__ == "__main__":
    unittest.main()