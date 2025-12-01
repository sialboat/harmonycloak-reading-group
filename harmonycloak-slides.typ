#import "@preview/touying:0.6.1": *
#import themes.simple: *

#show: simple-theme.with(aspect-ratio: "16-9")
#show link: underline

// = HarmonyCloak: Unlearnable \ Music Watermarking
// You can open this file using any text editor and follow along.
// One that I typically use is VSCode with the "Tinymist" extension
// installed. If you want, you can preview this text file by pressing
// "Command" + "K" and then "V" (assuming you have tinymist installed).
// You can also treat this as a regular text file as well.

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
  #set text(16pt)
  + Issue(s) at Large \
    a.) Automation \
    b.) Privacy \
    c.) The Arts \
    d.) Threat Model \
  + Preliminaries \
    a.) Psychoacoustics \
    b.) AI Overview \ //
    c.) Adversarial Machine Learning \
  + HarmonyCloak \
    a.) Design \
    b.) Harmony Cloak & MIDI/Audio \
    c.) Optimization \
    d.) Evaluation \

][
  #set align(center)
  #set text(18pt)
  Slides, Notes, References:
  // image
  #image("Images/qr-code.svg", width: 65%)
  // https://github.com/sialboat/harmonycloak-reading-group
  #link("https://tinyurl.com/genaudio-harmonycloak")[tinyurl.com/genaudio-harmonycloak]
]

// big picture idea: technology is rapidly advancing in development and it is becoming readily evident that
// we do not have enough safeguards in place to keep a healthy standard of life for all.
= Issue(s) at Large
#pause
\
technology is advancing fast enough to outpace our measures for the displaced
== Automation
// Automation is advancing at a rate that outpaces our fallback measures for the
// displaced. We see this everywhere.
#slide[
  #set text(18pt)
  #grid(
    columns: (auto, auto),
    column-gutter: 2em,
    [
      #set align(center)
      #block(above: 2pt, below: 2pt)[
        #set text(12pt)
        // https://allthatsinteresting.com/rust-belt#10
        #image("Images/task-displacement.jpg", width: 100%)
        _Task Displacement by Education_
      ]
      #block(above: 2pt, below: 2pt)[
        #image("Images/Self-Driving-Trucks.png", width: 60%)
        #set text(12pt)
        _Self Driving Trucks_
      ]
    ],
    [
      #set align(center)
      #block()[
        // https://onlinelibrary.wiley.com/doi/full/10.3982/ECTA19815
        #image("Images/labor-share-vs-robots.jpg", width: 100%)
        #set text(12pt)
        #set align(center)
        _Labor share & Automation_
      ]
      #block()[

        #image("Images/millionaire's-row.jpeg", width: 70%)
        #set text(12pt)
        _Millionaires' Row, Cleveland Ohio_
        // By the 1960s, the street that once rivaled Fifth Avenue as the most expensive address in the United States was a two-mile (3 km) long slum of commercial buildings and substandard housing.
        // https://en.wikipedia.org/wiki/Euclid_Avenue_(Cleveland)#East_30th_St.

        // i was going to use these but that was work for formatting
        // https://digitaleconomy.stanford.edu/wp-content/uploads/2025/11/CanariesintheCoalMine_Nov25.pdf
        //#image("AI-Exposure-Table.png", width: 90%)
      ]
    ],
  )
]

== Privacy
// With the advancement of technology comes the concern for data collection and
// usage
#slide[
  #grid(
    column-gutter: 2em,
    align: center,
    columns: (auto, auto),
    [ #block[
        // Think Edward Snowden but slightly worse
        #image("Images/palantir.png", width: 60%)
        #set text(12pt)
        _Government investment(s) in Surveillance_
      ]
      #block[
        #set text(12pt)
        #image("Images/False-positives-ai-security-camera.png", width: 60%)
        _AI-driven Surveillance False Positives_
      ]
    ],
    [
      #block[
        #image("Images/nyu-mishandling-data.png", width: 80%)
        #set text(12pt)
        _Poorly managing sensitive data_
      ]
    ],
  )
  #set text(18pt)
  // palantir
  // Benn Jordan FLock Camera false positives
  // NYU Data Breach
  // Smart Twitter Ragebait Bots (Benn Jordan)
]
== ... and Misinformation
#slide[
  #grid(
    columns: (auto, auto),
    column-gutter: 2em,
    align: center,
    [
      // apparently Twitter put out a change that had people state which location their account was based
      // and just about every single MAGA account you could imagine dropped from the face of the earth
      // (they were not based in America)
      // https://www.businessinsider.com/x-shows-locations-users-about-this-account-2025-11?op=1
      // Image from Benn Jordan's YouTube Channel
      #block[
        // https://www.techradar.com/computing/cyber-security/ai-impersonation-scams-are-sky-rocketing-in-2025-security-experts-warn-heres-how-to-stay-safe
        #image("Images/AI-impersonation.png", width: 60%)
        #set text(12pt)
        #set align(center)
        _AI Voice Deepfake Scams_
      ]
    ],
    // bear in mind this does not even go over the predatory social media algorithms that optimize consumption
    // that has been with us in our lives now
    [ #block[
        #image("Images/deepfake-swift.jpg", width: 60%)
        #set text(12pt)
        #set align(center)
        _This Deepfake Scandal_
      ]
      #block[
        #image("Images/twitter-bots.jpg", width: 60%)
        #set text(12pt)
        _Dead Internet Theory_
      ]
    ],
  )
]

// Suno x Warner Music Group
// Suno & Udio Copyright Infringement
// The Velvet Sundown
// Ghibli style robbery
== The Arts
#slide[
  #set text(18pt)
  #grid(
    columns: (auto, auto),
    column-gutter: 2em,
    align: center,
    [
      #image("Images/music-AI-lawsuit.jpg", width: 60%)
      #set text(12pt)
      _Everybody's Favorite Music Lawsuit_
      // don't even mention how flawed the streaming music model is. just go back to
      // piracy bruh
      #image("Images/spotify-drone-investments.jpg", width: 40%)
      #set text(12pt)
      _Spotify Drone Investments_
    ],
    [
      #image("Images/velvet-sundown.png", width: 80%)
      #set text(12pt)
      _Everybody's Favorite Human Band_
    ],
  )
]

== Threat Model
#slide[
  #set text(16pt)
  #pause
  *Attacker's Capability:* \
  - Scrape music from the internet to train Generative AI models
  - Copyright infringement, musicians do not get paid
  #pause
  *Defender's Objective:* \
  - Share music that Generative AI can't scrape
  - Must be high-fidelity
  #pause
  *Defender's Capabilities:* \
  - Computer
  - Can write and share songs
  - Knowledge of the generative model#footnote("in a black box setting, this is not the case. In a white box setting, this is.")
][
  #meanwhile
  #set align(center)
  #image("Images/style-transfer.jpg", width: 70%)
]

= Preliminaries

== Psychoacoustics
#grid(
  columns: (auto, auto),
  rows: (auto, auto),
  gutter: 3pt,
  [#set text(16pt)
    #pause Human hearing has a lower limit; a softest sound we\
    can possibly hear. This *absolute hearing threshold* \
    can be described mathematically for any frequency $f$:
    $
      T_H (f) = 3.64 dot (f / 1000)^(-0.8) - 6.5 dot e^( -0.6 (f/1000-3.3)^2 )
    $
    \
  ],
  grid.cell(rowspan: 2)[#image("Images/Frequency-Masking.png")],
  [
    #set text(16pt)
    #pause  Additionally, given two sounds $A$ and $B$ with similar \
    frequency $(f_A approx f_B)$, we say that $A$ *masks* $B$ if we \
    can hear $A$ but not $B$. #footnote()["formally this is known as Auditory Masking"]
    - *Temporal Masking* can also happen if $A$ and $B$ \
      are not played at the same time.

  //https://en.wikipedia.org/wiki/Auditory_masking
  //The math to calculate each specific threshold for each frequency depends on how loud
  //the "sound to mask" is in comparison with the "sound that masks"
  ],
)

== Generative AI Overview
#slide[
  #set text(18pt)
  // Generative AI. RNN, GAN, or Transformer-based models
  We have devised multiple different ways to create *Generative AI*; Artificial Intelligence that can
  tailor output responses based on user input.

  #pause
  - #underline("GAN") (_University of Montreal, 2014_): Two models work in parallel towards a unified goal.
    + *Generator:* Trained to generate a target output
    + *Discriminator:* Trained to determine if (1)'s target output is good or bad
  #pause
  - #underline("Diffusion Models") (_Stanford & UC Berkley, 2015_): Increasingly add or subtract noise given a prompt
    to achieve a desired output.
  #pause
  - #underline("Transformers") (_Google, 2017_): Prominent in NLP, each part of the input is given a fixed
    weight with respect to its relevance to the other parts of the input, which the model can then
    use to predict / generate a response. The T in ChatGPT stands for Transformer.

  #pause
  #set text(fill: red)
  In order to protect our audio, we must gaslight the clanker's learning process (described above).
]
== Generative Audio
#slide[
  //
  #set text(18pt)
  // From HarmonyCloak
  Some prominent Generative AI Architectures with respect to audio use *transformers* or *GANs*: #pause

  - #underline("MIDINet & MuseGAN"): Convolutional Neural Network GAN, trained on tons of "piano rolls"; outputs MIDI #pause
  - #underline("MusicBERT"): Extracts characteristics using a Transformer-based model. #pause
  - #underline("MusicLM"): Transformer-based model that generates audio from text (Consistent, Proprietary, Google) #pause
  - #underline("MusicGen"): Transformer-based model that generates audio from text (Flexible, Open Source, Meta)
]

== Adversarial Attacks & Machine Learning
#slide[
  // this is how we will gaslight the generative models
  #set text(18pt)
  To gaslight a generative model, we must make it think that something else is right. #pause

  We can achieve this using some kind of _error-maximizing noise_ #pause
  - Throw off the model's predictions $==>$ gaslighting #pause
  - Examples include Adversarial Attacks and Data Poisoning #pause
  - Useful for testing Robustness of a ML Model pause
  #pause

  However, just because a model's output is horribly inaccurate does not mean that it cannot
  continue to learn off of the given data. #pause

  *Unlearnable Example*: utilizes _error-minimizing noise_
  - Developed for #smallcaps("HarmonyCloak")
  - Trick the model into thinking "there is nothing left to learn" #pause $==>$ $#rect("gaslighting & data protection")$
]

= HarmonyCloak


== Design
#slide[
  #set text(18pt)

  #smallcaps("HarmonyCloak") must produce noise that does not degrade audio fidelity and effectively gaslights Generative Audio models.
  #pause
  So we must fulfill these four requirements for #smallcaps("HarmonyCloak") to properly generate "unlearnable examples" (UEs):
  #set text(22pt)
  \

  #pause
  + The noise must be masked by the song #pause
  + The noise must adapt _dynamically_ to the song's structure #pause
  + The noise must adapt _timbrically_ to the song's instruments #pause
  + The noise must not degrade under compression algorithms
]

== Creating Unlearnable Examples
#slide[
  #set text(18pt)
  Two possible scenarios: the white box and black box. \ \ #pause
  In a *white box* setting (we know what model they're using):
  + Split the song into small chunks (windowing) #pause
  + Generate the noise for each small chunk
  + Enjoy a cup of hot chocolate and complete your ACM Final Project

  #pause

  In a *black box* setting (we don't know what model they're using):
  + Split the song into small chunks (windowing) #pause
  + Figure out which model is used via Meta-Learning
    - Generate noise and run tests for multiple models
    - fine-tune each noise injection technique, pick the most optimal one (Meta-testing) #pause
  + Enjoy a cup of hot chocolate and read your DST Chapters
]

// it's important to know there's a distinction here:
// we train HarmonyCloak on *both* MIDI / Symbolic music
// and actual audio here

== HarmonyCloak with MIDI
#slide[
  Given MIDI as input: #pause

  + Split the MIDI file into small chunks / windows #pause
  + _determine what the spectrogram of the MIDI instrument looks like_ #pause
  + Generate noise for each chunks / windows for each local amplitude peak. #pause
    - No louder than the masking noise, no softer than the absolute hearing threshold. #pause
  + Enjoy a cup of hot chocolate and get a good night's rest.
]

== HarmonyCloak with Audio
#slide[
  #pause Given Audio as Input:

  + Split the audio file into small windows #pause
  + _Conduct a STFT to obtain dominant frequencies in dB SPL_ #pause
  + _After finding the relative dB SPL, _ introduce the noise component #pause
    - No louder than the masking noise, no softer than the absolute hearing threshold. #pause
  + _FFT your results back into the Time Domain_ #pause
  + Enjoy a cup of hot chocolate and work on your GenAudio Projects
]

== Loss Function and Optimizations
#slide[
  // this is also how we will gaslight the generative models
  #set text(16pt)
  #set align(center)
  When creating Generative Models, we want to _optimize_ our results such that the output of said is close
  enough to the target output that we envision it to achieve. #pause
  #set text(fill: red)
  We exploit this to do our gaslighting:
  #set text(fill: black)
  #pause
  $
    min_(theta) EE[min_delta cal(L)_(g e n) (f(x + delta))] + alpha sum_(m = 1)^M w^m || delta^m ||_2 "(generate the noise)"\
    "such that" \
    cal(H)(T_H) <= delta^m <= x^m forall m in {1, 2, ... M}
  $

  where

  $cal(H)(T_H(v)) = 127 dot 10^(1/4 log_(10)(T_H(v)) - L_U + 94) "(constrain the magnitude, given note velocity fo for audior MIDI)"$


  $ cal(H)(T_H(v^m_t)) <= delta^m_t <= cal(M)(x^m_t), forall t, m "(windowize it with a sigmoid function for audio)" $
  #pause
  _in other words, this is how we mathematically describe the high level stuff we have been building up to_

  // From the Notes that I took: (I have no idea what most of this means but from the infrences and the places
  // Perplexity pointed me, the slides are the deductions that I can make based off of this stuff)
  // GAN
  // Find the smallest possible parameter values for the Discriminator(D) to fool the generator (G) that it is fake.
  //$
  //min_(G) max_(D) [(min_(delta) EE_(x + delta)[log D(x + delta)] + EE_z [log(1 - D(G(z)))]]
  //$
  // AutoRegressive Models / Transformers:
  // Find the smallest possible (outer min_theta) parameters for the optimal delta (noise parameter) that minimizes
  // the model's loss
  //$
  //min_(theta) min_(delta) - log(Pi_(t = 1)^(T) f(x_t|x_(t-1) + delta_(t-1), ... x_(t-p) + delta_(t-p) : theta))
  //$
  // From my notes, this is the black-box optimization process:
  // $
  // min_theta EE [min_delta (sum_(q = 1)^(Q - 1) cal(L)^q_(g e n) f(x + delta})] + alpha sum_(m = 1)^M w^m || alpha^m||_2 \
  //  "s.t" \
  //  cal(H)(T_H) <= alpha^m <= x^m forall m in {1, 2, ... M} $
  //  \ \ \
  // 
  // - $f(dot)$ is the generative model
  // - $cal(L)_(g e n)(dot)$ is generative loss of model
  // - $T_H$ is absolute human hearing threshold
  // - $alpha$ is scaling coefficient
  // - $w^m$ is $1 - "ratio of note velocity of track" m "to cumulative note velocity of all tracks"$
  //   - for balancing noise intensity added to each instrumental track
  // - $cal(H)(dot)$ converts dB SPL threshold to note velocity
  // $
  //   cal(H)(T_H(v)) = 127 dot 10^(1/4 log_(10)(T_H(v)) - L_U + 94)
  // $
  // - $L_U$ is magnitude of lienar transfer function normalized at 1kHz.
  // - inner minimization tries to find noise that minimizes overall generative loss on unlearnable music
  // - outer minimization adjusts model parameters $theta$ (or the generator if GAN) to minimize generative model loss
  // - regularizer term attempts to minimizes overall amplitude of added noise through reducing $L_2$ norm of noise.
  //   - constraints in the $cal(H)(T_H)$ equation above makes sure that noise gets constrained in the masked regions, sandwiched between the music amplitude and the threshold for hearing.

  // Use Projected Gradient Descent (PGD) as first-order optimization method to solve constrained inner minimization problem.
  // - Apply Stochastic Gradient Descent (SGD) for outer minimizastion

  // Window based Strategy for Temporal Dynamics \
  // - Divide the music bar $x$ into short non-overlapping windows during optimization. Each has length $l_w$.
  //   - for each window $x_t$, find frequency $v^m_t$ of dominant musical note for every track (track $m$) based on cumulative musical note velocities, set defensive noise $delta^m_t$ for window $t$ and track $m$ to same frequency $v^m_t$.
  //   - if there is no dominant musical note, no noise will be introduced. Noise is introduced uniquely for each instrumental track, so the strategy hinges upon creating defensive noises for each track.

  // Apply to windows using a `sigmoid()` function to constrain $delta^m_t$ to the following range for all $t$ windows $m$ tracks:
  // $
  //   cal(H)(T_H(v^m_t)) <= delta^m_t <= cal(M)(x^m_t), forall t, m
  // $
  // - $v^m_t$ is dominant frequency, $cal(M)(dot)$ is velocity of dominant musical note at bar $x^m_t$



]

== Evaluation
#slide[
  // Include QR Code to website here Silas
  #grid(
    columns: (auto, auto),
    rows: (auto, auto),
    [
      #set align(horizon)
      #image("Images/Results.png", width: 60%)
    ],
    [
      #set align(horizon)
      #image("Images/User Study.png")
    ],
    grid.cell(colspan: 2)[
      #pause
      \
      #set text(18pt)
      #set align(center)
      #link("HarmonyCloak website")[https://mosis.eecs.utk.edu/harmonycloak.html]
    ],
  )
]

= thanks\\

// further reading can be found in the HarmonyCloak paper
// also check out Benn Jordan's video on AI Music Malware,
// as he describes in detail how adversarial noise attacks
// can be used to further gaslight Generative Audio models.
// Similarly, his videos on hacking into Surveillance Cameras
// can be another watch to show applications outside of Music
// Technology. Hopefully the source is useful for those who want to read it!
// - Silas
