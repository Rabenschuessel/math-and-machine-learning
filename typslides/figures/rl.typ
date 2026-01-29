#import "@preview/cetz:0.4.2"
#cetz.canvas({
  import cetz.draw: *
  rect((-2.5, 2), (2.5, 4),  radius: 0.25, name:"agent")
  rect((-2.5, 0), (2.5, -2), radius: 0.25, name:"env")

  set-style(mark: (end: ">"))
  line("agent.east", (rel:(1.5, 0)) , (rel: (0, -4)), "env.east"  , name: "action")
  line("env.west"  , (rel:(-1  , 0)), (rel: (0, 4)) , "agent.west", name: "state" )
  line("env.west"  , (rel:(-1.5, 0)), (rel: (0, 4)) , "agent.west", name: "reward")

  content(("agent.north", 50%, "agent.south"), [Agent])
  content(("env.north", 50%, "env.south")    , [Environment])

  content((rel: (2, 0)  , to:("action.start", 50% , "action.end")) , angle: 270deg, [Action])
  content((rel: (-2, 0)  , to:("state.start", 50% , "state.end")) , angle: 90deg, [State])
  content((rel: (-0.6, 0), to:("reward.start", 50%, "reward.end")), angle: 90deg, [Reward])
})
