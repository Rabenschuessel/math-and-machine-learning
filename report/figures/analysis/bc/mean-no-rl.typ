#set table(stroke: none)
#figure(table(
  columns: 5,
  [dataset],  [epochs], [fc]   , [cnn]  , [resnet],
  table.hline(),                         
  [puzzle],        [1], [0.507], [0.400], [0.830],
  [],             [10], table.cell(fill: aqua)[0.530], table.cell(fill: aqua)[0.446], table.cell(fill: aqua)[0.925],
  table.hline(start:1),         
  [grand masters], [1], [0.390], [0.396], [0.797],
  [],             [10], [0.418], [0.434], [0.870],
),
caption: [
  Expected proportion of points for models not using reinforcement learning.
  (best model of architecture highlighted).
]
) <table-pretraining-no-rl>

