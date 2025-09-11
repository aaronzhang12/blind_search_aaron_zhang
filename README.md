[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zfxHeKS3)
# assignment-1-blind-search

# TASK 2

rewrite to make sure all cells are visited

Maze Generation is very similar to maze solver. We start at the same start and goal states. The state space involves all cells on the board. We start with a completely walled off grid except for the start and goal states, and run either bfs or dfs to break these walls (without returning to open cell to avoid path crossing). The transition states are all the unbroken walls 4-side adjacent to the current cell. Eventually, one of these wall breaking paths will reach the end. All other explored paths are dead-ends in the maze. 

# TESTS

Collaborators (if any):

AI Use Description:

You must acknowledge use here and submit transcript if AI was used for portions of the assignment
