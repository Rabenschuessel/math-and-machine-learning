#import "@preview/basic-report:0.3.0": *
#import "@preview/subpar:0.2.2"
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()
#codly(zebra-fill: none)

#show: it => basic-report(
  doc-title: "Exploration of Rewards Shaping and Immitation Learning in Chess",
  author: "Jakob Lambert-Hartmann",
  language: "en",
  compact-mode: false,
  show-outline: false,
  heading-font: "Fira Math",
  it
)

#set text(size: 11pt, font: "Fira Sans")
#set page(margin: (x: 2.5cm, top:4cm, bottom:3cm))
#set page(footer:  grid.cell(colspan: 2, line(length: 100%, stroke: 0.5pt)))
#set par(justify: true)
#set heading(numbering: none)
#show heading: set text(weight: "bold")
#show heading.where(level: 4): it => box(it.body) 

#outline(depth: 3)
#pagebreak()

