[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zfxHeKS3)
# assignment-1-blind-search

# TASK 2

rewrite to make sure all cells are visited

Maze Generation is very similar to maze solver. The state space involves all cells on the board. We start with a completely walled off grid and randomize the start state location, and run either bfs or dfs to break these walls (without returning to open cell to avoid path crossing). The transition states are all 4-side adjacent cells to the current cell for which the corresponding walls have not been broken (a path not explored). The goal state is achieved when we have explored all cells, with one valid path. All other explored paths or subsets of explored paths are dead-ends in the maze. 

# TESTS

I wrote extra functions to call in different tests. The first is that a path is valid, meaning that each state in the path is truly a sucessor. It also checks that the length of the path is the length of the set of the path, in order to check no duplicate states in the path. Additionally, I wrote a function that validates stats for either algorithm. It ensures the proper path length is returned, and creates bounds on states_expanded and max_frontier_size, ensuring the lower bound is 0 exclusive, and the upper bound is the area of the grid, or the number of cells. It also upper bounds the path length to the area as well. 

I ran these two functions on all maze sizes in both bfs and dfs tests, and checked lower bounds on path length as well. Additional tests include testing that no path returned when no path exists, that bfs returns a shorter path than dfs, that bfs returns the optimal or one of the optimal paths, and that we never go out of bounds on the maze. 

Finally, I created a graph with a short and long path to the solution to test the proper queue/stack structure of bfs, that bfs would get the shorter and dfs would get the longer path. 

Collaborators (if any):

AI Use Description:

Used AI to generate small Maze setup and also the directed graph setup. 

You must acknowledge use here and submit transcript if AI was used for portions of the assignment
