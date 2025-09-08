from typing import List, Tuple, Dict, Optional
from collections import deque  
from search_problem import SearchProblem, State
from maze import Maze

class PathNotFoundError(Exception):
    def __init__(self, message="Path not found"):
        self.message = message
        super().__init__(self.message)
        
def bfs(problem: SearchProblem[State]) -> Tuple[Optional[List[State]], Dict[str, int]]:
    """
    Performs Breadth-First Search (BFS) on the given problem.

    Args:
        problem (SearchProblem[State]): The search problem to solve.

    Returns:
        Tuple[Optional[List[State]], Dict[str, int]]:
            - A list of states representing the solution path, or None if no solution was found.
            - A dictionary of search statistics, including:
                - 'path_length': The length of the final path.
                - 'states_expanded': The number of states expanded during the search.
                - 'max_frontier_size': The maximum size of the frontier during the search.
    """
    # TODO: Update these values
    stats = {"path_length": 0, "states_expanded": 0, "max_frontier_size": 0} 

    q = deque()
    seen = set()
    start = problem.get_start_state()

    if not start:
        return (None, stats)

    q.append(start)
    seen.add(start)
    stats["states_expanded"] += 1
    stats["max_frontier_size"] += 1
    predecessor_dict = {}
    
    while q:
        state = q.popleft()
        if problem.is_goal_state(state):
            path = reconstruct_path(predecessor_dict, state, problem)
            stats["path_length"] = len(path)
            return (path, stats)
        for neighbor in problem.get_successors(state):
            if neighbor not in seen:
                q.append(neighbor)
                seen.add(neighbor)
                stats["states_expanded"] += 1
                stats["max_frontier_size"] = max(stats["max_frontier_size"], len(q))
                predecessor_dict[neighbor] = state
    
    return (None, stats)
    


    raise PathNotFoundError

def dfs(problem: SearchProblem[State]) -> Tuple[Optional[List[State]], Dict[str, int]]:
    """
    Performs a depth-first search (DFS) on the given search problem.

    Args:
        problem (SearchProblem[State]): The search problem to solve.

    Returns:
        Tuple[Optional[List[State]], Dict[str, int]]:
            - A list of states representing the solution path, or None if no solution was found.
            - A dictionary of search statistics, including:
                - 'path_length': The length of the final path.
                - 'states_expanded': The number of states expanded during the search.
                - 'max_frontier_size': The maximum size of the frontier during the search.
    """
    # TODO: Update these values
    stats = {"path_length": 0, "states_expanded": 0, "max_frontier_size": 0}

    q = deque()
    seen = set()
    start = problem.get_start_state()

    if not start:
        return (None, stats)

    q.append(start)
    seen.add(start)
    stats["states_expanded"] += 1
    stats["max_frontier_size"] += 1
    predecessor_dict = {}
    
    while q:
        state = q.pop()
        if problem.is_goal_state(state):
            path = reconstruct_path(predecessor_dict, state, problem)
            stats["path_length"] = len(path)
            return (path, stats)
        for neighbor in problem.get_successors(state):
            if neighbor not in seen:
                q.append(neighbor)
                seen.add(neighbor)
                stats["states_expanded"] += 1
                stats["max_frontier_size"] = max(stats["max_frontier_size"], len(q))
                predecessor_dict[neighbor] = state
    
    return (None, stats)

    raise PathNotFoundError

def reconstruct_path(path: Dict[State, State], end: State, problem: SearchProblem[State]) -> List[State]:
    """
    Reconstructs the path from the start state to the given end state.
    
    This function traces back through the parent mapping to build the complete path
    from start to goal. The path is initially built in reverse order (goal to start)
    and then reversed to give the correct order.

    Args:
        path (Dict[State, State]): A dictionary mapping each state to its predecessor in the search.
        end (State): The goal state to trace back from.
        problem (SearchProblem[State]): The search problem to solve.

    Returns:
        List[State]: The reconstructed path from the start state to the goal state.
    """
    reverse_path = []
    while end != problem.get_start_state():
        reverse_path.append(end)
        end = path[end]
    reverse_path.append(problem.get_start_state())
    reverse_path.reverse()
    return reverse_path

############### SANDBOX ###############
def main() -> None:
    # Initialize the maze and generate it based on given dimensions
    print("Generated Maze:")
    width, height = 30, 30
    maze = Maze(width, height)
    
    # Run BFS and DFS to find paths
    print("BFS Path:")
    bfs_path, bfs_stats = bfs(maze)
    print(f"BFS found path: {bfs_path}")
    print(f"BFS stats: {bfs_stats}")
    maze.visualize_maze(path=bfs_path, algorithm_name="bfs")
    
    print("DFS Path:")
    dfs_path, dfs_stats = dfs(maze)
    print(f"DFS found path: {dfs_path}")
    print(f"DFS stats: {dfs_stats}")
    maze.visualize_maze(path=dfs_path, algorithm_name="dfs")


if __name__ == "__main__":
    main()
