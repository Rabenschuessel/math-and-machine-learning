#set table(stroke: none)

#figure(
  stack(dir:ltr, spacing: 0.5cm,
  
  table(
    columns: 4,
    
    [rewards], [fc]   ,  [cnn], [resnet],
    table.hline(),
    [none],    [0.461],[0.419], table.cell(fill: aqua)[0.855],
    [r_0],     table.cell(fill: aqua)[0.520],table.cell(fill: aqua)[0.547], [0.404],
    [r_1],     [0.504],[0.532], [0.405],
    [r_2],     [0.507],[0.486], [0.407],
  ),
  table(
  columns: 4,
  [rewards], [fc],    [cnn], [resnet],
  table.hline(),
                          
  [r_0],         [0.289],  table.cell(fill: aqua)[0.266], table.cell(fill: aqua)[0.408],
  [r_1],         [0.307],  [0.235], table.cell(fill: aqua)[0.408],
  [r_2],         table.cell(fill: aqua)[0.335],  [0.251], [0.399],
)
),


caption: [
  Expected points by reward set. 
  *Left:* across all models.
  *Right:* across models wihtout any supervised pretraining. 
]
)<table-rl>
