#set table(stroke: none)


#table(
  columns: 5,
  [], [architecture], [cnn], [fc], [resblock],
  [train], [rewards], [], [], [],

  [gm_1], [none], [0.396], [0.390], [0.797],
  [], [r_0], [0.605], [0.483], [0.411],
  [], [r_1], [0.650], [0.523], [0.409],
  [], [r_2], [0.572], [0.494], [0.424],
  [gm_10], [none], [0.434], [0.418], [0.870],
  [], [r_0], [0.714], [0.494], [0.387],
  [], [r_1], [0.694], [0.538], [0.398],
  [], [r_2], [0.654], [0.526], [0.399],
  [pz_1], [none], [0.400], [0.507], [0.830],
  [], [r_0], [0.478], [0.697], [0.410],
  [], [r_1], [0.564], [0.631], [0.410],
  [], [r_2], [0.433], [0.623], [0.409],
  [pz_10], [none], [0.446], [0.530], [0.925],
  [], [r_0], [0.672], [0.637], [0.403],
  [], [r_1], [0.515], [0.521], [0.403],
  [], [r_2], [0.520], [0.558], [0.403],
  [untrained], [none], [nan], [nan], [nan],
  [], [r_0], [0.266], [0.289], [0.408],
  [], [r_1], [0.235], [0.307], [0.407],
  [], [r_2], [0.251], [0.335], [0.399],
)

best models: 
cnn gm_10 r_0
fc pz_1 r_0
resblock gm_10 none


#table(
  columns: 4,
  [architecture], [cnn], [fc], [resblock],
  [train], [], [], [],

  [gm_1], [0.556], [0.472], [0.510],
  [gm_10], [0.624], [0.494], [0.513],
  [pz_1], [0.469], [0.615], [0.515],
  [pz_10], [0.538], [0.562], [0.533],
  [untrained], [0.251], [0.310], [0.405],
)

fc is better for one puzzle epoch compared to 10.
maybe because it becomes overconfident in it's tactical moves, where there are none. 
This is however not seen in the other models.
Overall there is also no clear winner between training on masters or puzzles.


#table(
  columns: 4,
  [architecture], [cnn], [fc], [resblock],
  [rewards], [], [], [],

  [none], [0.419], [0.461], [0.855],
  [r_0], [0.547], [0.520], [0.404],
  [r_1], [0.532], [0.504], [0.405],
  [r_2], [0.486], [0.507], [0.407],
)


for fc and cnn r_0 is best
for resblock any reinforcement training performs significantly worse than no rl.
