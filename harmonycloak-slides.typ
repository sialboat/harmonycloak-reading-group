#import "@preview/touying:0.6.1": *
#import themes.simple: *

#show: simple-theme.with(aspect-ratio: "16-9")
#show link: underline

// = HarmonyCloak: Unlearnable \ Music Watermarking
// #show title: set text(size: 48pt)
// #show title: set align(center)

#slide[
  #set text(36pt)
  #set align(center)
  = HarmonyCloak
  #set text(27pt)
  _Gaslighting Generative Audio Models_

  #set text(18pt)
  Silas Wang ; NYU GenAudio
]
 == Agenda
#slide[
  #set text(18pt)
+ Issue(s) at Large \
  a.) Automation \
  b.) Privacy \
  c.) The Arts \
+ Preliminaries \
  a.) Psychoacoustics \
  b.) AI Overview \ //
  c.) Noise-based Attacks
+ HarmonyCloak \
  a.) Threat Model \
  b.) Design \
  c.) Evaluation
][
  #set text(18pt)
  Slides, Notes, and References:
  // image
  #image("qr-code.svg", width: 60%)
  // https://github.com/sialboat/harmonycloak-reading-group
  or consult #link("https://tinyurl.com/genaudio-harmonycloak")[tinyurl.com/genaudio-harmonycloak]
]

= Issue(s) at Large 
== Automation
// Automation is advancing at a rate that outpaces our fallback measures for the
// displaced. We see this everywhere.

#slide[
  #set text(18pt)
  #grid(
    columns: (auto, auto), column-gutter: 2em,
    [
      #set align(center)
      #block(above: 2pt, below: 2pt,)[
        #set text(12pt)
        // https://allthatsinteresting.com/rust-belt#10
              #image("task-displacement.jpg", width: 100%)
        _Task Displacement by Education_
     ] 
      #block(above: 2pt, below: 2pt,)[
        #image("Self-Driving-Trucks.png", width: 60%)
        #set text(12pt)
        _Self Driving Trucks_
      ]
    ],
  [
          #set align(center)
    #block()[
      // https://onlinelibrary.wiley.com/doi/full/10.3982/ECTA19815
      #image("labor-share-vs-robots.jpg", width: 100%)
      #set text(12pt)
      #set align(center)
      _Labor share & Automation_
    ]
    #block()[

        #image("millionaire's-row.jpeg", width: 70%)
        #set text(12pt)
        _Millionaires' Row, Cleveland Ohio_
        // By the 1960s, the street that once rivaled Fifth Avenue as the most expensive address in the United States was a two-mile (3 km) long slum of commercial buildings and substandard housing.
        // https://en.wikipedia.org/wiki/Euclid_Avenue_(Cleveland)#East_30th_St.
 
      // i was going to use these but that was work for formatting
      // https://digitaleconomy.stanford.edu/wp-content/uploads/2025/11/CanariesintheCoalMine_Nov25.pdf
      //#image("AI-Exposure-Table.png", width: 90%)
    ]
  ]
  )
]

== Privacy 
#slide[
  #set text(18pt)


]

== The Arts 
#slide[
  #set text(18pt)

]

#set text(27pt)
= Preliminaries


== Psychoacoustics
#grid(
  columns: (auto, auto),
  rows: (auto, auto),
  gutter: 3pt,
  [#set text(16pt)
  Human hearing has a lower limit; a softest sound we\
  can possible hear. This *absolute hearing threshold* \
  can be described mathematically for any frequency $f$:
  $
    T_H (f) = 3.64 dot (f / 1000)^(-0.8) - 6.5 dot e^( -0.6 (f/1000-3.3)^2 )
  $
  \
  ],grid.cell(rowspan: 2)[#image("Frequency-Masking.png"
    )],
  [
    #set text(16pt)
    Additionally, given two sounds $A$ and $B$ with similar \
    frequency $(f_A approx f_B)$, we say that $A$ *masks* $B$ if we \
    can hear $A$ but not $B$.
    - *Temporal Masking* can also happen if $A$ and $B$ \
      are not played at the same time.
  ]

 )

== AI Overview

== Generative AI Models

== Noise-Based Attacks

= HarmonyCloak

== Threat Model
#slide[
  #set text(16pt)
  *Attacker's Capability:* \
    - Scrape music from the internet to train Generative AI models
    - Copyright infringement, musicians do not get paid
  *Defender's Objective:* \
    - Share music that Generative AI can't scrape
    - Must be high-fidelity
  *Defender's Capabilities:* \
    - Computer
    - Can write and share songs
    - Knowledge of the generative model#footnote("in a black box setting, this is not the case. In a white box setting, this is.")
][

]


== Design
#set text(18pt)
#smallcaps("HarmonyCloak") must produce noise that does not degrade audio fidelity and effectively gaslights Generative Audio models.

#set text(22pt)
=== Requirements:
+ The noise must be masked by the song
+ The noise must adapt _dynamically_ to the song's structure
+ The noise must adapt _timbrically_ to the song's instruments
+ The noise must not degrade under compression algorithms
== Evaluation
