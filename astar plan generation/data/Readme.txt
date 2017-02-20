dataset1
----------
input (map, start, end)
output(path) same as size of map with clamping to goal at the end


dataset2
------------
input [map, goal, current]
output [action category]

action categories:
[1,0,0,0];%top
[0,1,0,0]left 
[0,0,1,0]bottom 
[0,0,0,1]right 
