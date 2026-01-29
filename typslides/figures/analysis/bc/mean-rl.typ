#set table(stroke: none)

#table(
  columns: 5,
  [dataset],   [epochs], [fc]   , [cnn]  , [resnet],
  table.hline(),                  
                                  
  [none],            [], [0.310], [0.251], [0.405],
  table.hline(start: 1),          
  [puzzle],         [1], table.cell(fill: aqua)[0.615], [0.469], [0.515],
  [],              [10], [0.562], [0.538], table.cell(fill: aqua)[0.533],
  table.hline(start: 1),          
  [grand masters],  [1], [0.472], [0.556], [0.510],
  [],              [10], [0.494], table.cell(fill: aqua)[0.624], [0.513],
) 
