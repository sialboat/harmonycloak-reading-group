= HarmonyCloak

== Presentation
- agenda:
- Issues with data
  - disrespect of art within industries
  - disrespect of privacy
  - lack of ownership
  - surveillance
  - lack of any reason to care
- Watermarking in AI overview
- HarmonyCloak

== Notes

=== Issues at Large
// disrespect of privacy
// disrespect of art within industries
// lack of ownership
// surveillance
// leadership / politicians are lazy

=== Watermarking in Artificial Intelligence

=== HarmonyCloak
*Introduction:*
- DALL-E, Chat-GPT, etc. has revolutionized various domains.
- Music generative AI brings concern of unauthorized exploitation of coprighted music
  - can lead to financial loss and undermine artistic integrity
- we need safeguards for this shit so generative AI does not continue to pose a threat to artists

_Prior research in AI Data Protection:_
- many different attempts to make data samples unlearnable.
  - error-minimizing noise into data (degrading performance of models trained on said data)
  - traditional data-encryption and differential privacy methods
  - _unlearnable examples (UEs)_ do not compromise data quality for normal usage and also fucks with AI models
- most of these assume whitebox setting (all parameters are available)
- distance-maximizing noise to substantially shift image examples within feature space (good idea; but data isn't 100% unlearnable)

_Challenges_
+ Generative models aim to learn and replicate complex patterns and structures on their own.
  - generative models operate differently from classification models so they need specialized techniques based on generative model characteristics
+ Lack of clear ground truth for un-learnability
  - how do we assess how un-learnable something is?
+ Music is complicated! How do we generate unlearnable music whilst preserving its many different properties?

#smallcaps("HarmonyCloak")
- defensive framework to render unlearnable examples in music so musicians can safeguard their own work without help from others.
  - focuses on instrumental music, varied rhythmic structures, instrumentation.
- incorporate _error-minimizing_ noise into music training samples
  - protective measure to prevent useful data extraction.
- modifications remain imperceptible to human ear through _time-dependent_ optimization constraints
- account for Music Production process (translatable across various audio formats like mp3, wav, etc.)

TLDR:
- They claim #smallcaps("HarmonyCloak") is the first to use imperceptible error-minimizing noise to music that does not affect perceptual quality
- They claim that framework is beyond typical $L_p$-norm based / psychoacoustic hiding based methods
- Black-box and White-box approaches have both been developed to generate inperceptible noises (widely applicable)
- Extensive experiments have been done.

*Preliminaries*

_Unlearnable Examples_
Models learn when there is a gap between its current understanding and the data it encounters.
- quantified by loss generated from each data sample.
- UE(s) conventionally for classification models are based on how each data sample can be modified so classification loss is close to 0
- _zero loss_ gaslights model: nothing to learn from examples that are given.

Given data input $x$ with label $y$, we can generate _error-minimizing noise_ $delta$ by solving
$
  "arg" "min"_theta space EE_(x, y)[min_delta (cal(L)(f(x + delta), y)] space "s.t" space ||delta||_p <= epsilon
$
- $f$ is model, $cal(L)$ is cross-entropy loss, noise magnitude bounded by $||delta||_p$.
- to put it in context for generative models, we have to change things (this is for classification models)

_Comparison of NOise-Based Attacks/Defenses_
- _Error-maximizing noise_ / adversarial attacks maximize prediction errors in AI during testing phase
  - Adversarial Training allows to enhance model robustness by integrating bad examples (min-max optimization problem)
- _Data poisoning attacks_ degrade model's performance by tampering with training data
- Backdoor attacks are subset of data poisoning attacks that embed triggers in training data that mislead the model to incorrectly respond to trigger patterns.
- _Error-maximizing noise_ on training samples for data poisoning is really effective, but applying error-maximizing noise to training samples doesn't stop a model from learning.
- _Error-minimizing noise_ approach for unlearnable examples; error-minimizing noise through min-min optimization process;
  - gaslight model into thinking there is nothing that can be learned.

_Representation of Music Signals_
- MIDI for music composition, specifying which notes should be played. Also Audio exists.
- Some generative models like Google MuLan and MusicLM use audio directly to generate sound rather than just MIDI.
  - lead to high-fidelity music generation and timbre subtleties that MIDI doesn't always replicate.
- #smallcaps("HarmonyCloak") needs to accomodate for both MIDI and Audio
- MIDI is matrix with values ranging from 0 to 1, each element represents presence or absence of notes at different time steps.
  - e.g multi track piano roll for $M$ tracks can be represented as tensor $x in [0, 1]^(R times S times M)$. $R$ is time steps in a bar, $S$ is number of available note candidates.
  - Raw audio interpretation can either be converted from MIDI or represnted as $x in [-1, 1]^N$ for $N$ samples.

_Aspects of Music_
- research exists on audio machine learning attacks for imperceptible noise injection, but primary target of these papers is speech data to fuck with speech recognition systems.
- Employ $L_p$ norm or psychoacoustic hiding properties to constrain noise
  - may not be directly applicable for music.
- music is multi-voice / polyphonic; extra layer on top of acoustic and perceptual complexities.
  - [39] provides human-in-the-loop attack to create adversarial music to evade copyright detection, but requires a lot of human effort and effectiveness depends on music complexity.
- No existing research considers audio streaming technology alongside the other aspects of music.

To ensure Perceptual quality of music stays the same when injecting noise, must demand more nuanced and low-effort approach.

_Psychoacoustic Modeling_
- Sounds falling below certain psychoacoustic hearing thresholds (absolute hearing threshold, etc.) become imperceptible.
Hearing threshold:
$
  T_H (upsilon) = 3.64 times ((upsilon) / 1000)^(-0.8) - 6.5 times e^(-0.6 times (upsilon/1000 - 3.3)^2)
$
Frequency masking: loud sounds overwhelm softer sounds
- critical bandwidth around masker frequency; dropoff exists
- masking can also happen when masker and maskee sounds aren't played simultaneously (temporal masking)
- louder sounds can mask quieter ones that occur before (pre-masking) or after (post-masking) occurrence.

_MP3 Lossy Compression_ - combines audio compression tech and psychoacoustic fuckshit for optimal balance between
audio fidelity and file size. Good for music distribution.

Masking to noise ratio (MNR) can be calculated, enables encoder to allocate fewer bits to masked regions (compressing audio to smaller file size)
- #smallcaps("HarmonyCloak") has to work on mp3s.

_Deep Music Generation_
- Early attempts: RNN / Recurrent Neural Networks
- Recent advancements: GAN / Generative Adversarial Networks and Transformers.
  - MIDINet and MuseGAN use CNN-based GAN for music generation
  - SeqGAN uses RNN GAN framework (generative model is based off a stochastic policy in reinforcement learning)
- Transformers have gotten to be really effective, so we use NLP and Computer Vision for Music Generation, each with encoding strats for capturing complex audio information
- MusicBERT uses OctupleMIDI encoding for granular MIDI event representation, SymphonyNet uses Multitrack multiinstrument MMP Repetable encoding for more complex source material.
- MusicLM: large foundation model for high fidelity music based on textural descriptions
  - can translate complex prompts into music compositions.
  -3 key models: SoundStream for acoustic tokens, w2v-BERT for semantic tokens, MuLan for conditioning
- 2 stages of MusicLM: Semantic modeling stage (learns to map MuLan audio tokens to Semantic tokens)
  - acoustic modeling stage where tokens are predicted based on previously generated tokens and input text.
- MusicGen: auto-regressive Transformer model operating over EnCodec tokenizer. Codebooks sampled at 50Hz

*Threat Model & UEs for Gen. AI*
- Attacker's capabilities (AI companies / model owners): can scrape music data from Internet or Music streaming platforms to train GenAI models. Leads to copyright infringements, harms musicians.
  - probably has advantages and capabilities (unrestricted access to training dataset and model params)
  - Ability to perform adaptive attack strats for learning the unlearnable
- Defender's Objectives (you, the musician) would like to render your piece inaudible to Generative Audio models.
  - become big challenge for AI company to learn from music.
  - concurrently wants to ensure UEs resemble original music whilst indistinguishable for non-human listeners.
- Defender's Capabilities: Defender has some kind of computer
- Defender can convert compositions into mp3 for distribution and full access to all project files and songs
- Defender has no access to model / training data (in a white bo setting, this would be different. defender must know how to use generative model to train on music data)
- in a black box setting, defender must operate without knowledge of what generative model is used

*UEs for Generative AI*
- Generative models like Transformers and GANs behave differently

Unlearnable Examples from Autoregressive Models \
- For autoregressive models like Transformers, we attempt to minimize negative log liklihood (NLL) of target sequence given predicted / partially generated sequence.
  - This is core for stuff like MusicLM and SymphonyNet. To reduce information that can be extracted by the models, introduce perturbation $delta$ to trick the model into thinking the current sequence is flawless.

For sequence $X_(1:N)$ we use the _min-min_ objective function to achieve this:
$
  min_(theta) min_(delta) - log(Pi_(t = 1)^(T) f(x_t|x_(t-1) + delta_(t-1), ... x_(t-p) + delta_(t-p) : theta))
$
- $x_t$ is predicted vbalue in sequence at given time $t$.
- $p$ is autoregressive order

after optimizing for optimal noise, unlearnable data ($x + delta$) can be used to train autoregressive model. _the result is that the model will no longer generalize on samples since noise loses the learning capability_

Unlearnable Examples for GAN-based Generative Models \
- Generative Adversarial Networks (MuseGAN) are different than autoregressives since they capture statistical properties of training data and generate samples resembling the training distribution.
- GAN contains two parts:
+ Generator network $(G)$. Aims to minimize loss through creating more realistic data
+ Discriminator network $(D)$. Aims to maximize loss by figuring out how to better discern real from fake data.

Minimize generator loss by adding $delta$ imperceivable noise
onto sample to discourage discriminator from properly classifying sample:
$
  min_(G) max_(D) [(min_(delta) EE_(x + delta)[log D(x + delta)] + EE_z [log(1 - D(G(z)))]]
$
- $x$ is a data sample
- $z$ is a random noise input
- After determining optimal noise and feeding in unlearnable data $x + delta$ into GAN, the loss reaches a minimum value during initial training rounds, suppressing informative knowledge of data.

*HarmonyCloak*
_Methodologies_

Design Rationale \
Four key aspects to tailor objective functions for creating unlearnable music examples:
- Concealing noiss with Music via Frequency Masking
  - Noise should be subtle and within critical bandwidth of the masking music tunes (frequencies must be close to musical notes)
- Dynamic Variations and Temporal Masking
  - Dynamic range and volume changes $==>$ noise must adapt over time and be close to the masking music, especially when they can't be played together.
- Track-Specific Noise Tailoring
  - defensive noises must be tailored for each instrumental (tempo, frequency range, timbre, etc. must be considered)
- Remaining Effectiveness under Lossy Compression
  - Noise must survive compression algorithms. So noise must not surpass dominant musical tones

Constrained Optimization Problem \
- We can use this to solve for optimal noise.

Given one bar of multi track music sample $x = [x^1, x^2, ... x^M]$ for $M$ tracks, imperceptible defensive noise is injected into each individual track #strong($delta$) $= [delta_1, delta_2, ... delta_M]$, giving us $x^i + delta_i$ for every track.

So we get the bilevel optimization problem:

$
  min_(theta) EE[min_delta cal(L)_(g e n) (f(x + delta))] + alpha sum_(m = 1)^M w^m || delta^m ||_2, \
  "s.t" \
  cal(H)(T_H) <= delta^m <= x^m forall m in {1, 2, ... M}
$
- $f(dot)$ is the generative model
- $cal(L)_(g e n)(dot)$ is generative loss of model
- $T_H$ is absolute human hearing threshold
- $alpha$ is scaling coefficient
- $w^m$ is $1 - "ratio of note velocity of track" m "to cumulative note velocity of all tracks"$
  - for balancing noise intensity added to each instrumental track
- $cal(H)(dot)$ converts dB SPL threshold to note velocity
$
  cal(H)(T_H(v)) = 127 dot 10^(1/4 log_(10)(T_H(v)) - L_U + 94)
$
- $L_U$ is magnitude of lienar transfer function normalized at 1kHz.
- inner minimization tries to find noise that minimizes overall generative loss on unlearnable music
- outer minimization adjusts model parameters $theta$ (or the generator if GAN) to minimize generative model loss
- regularizer term attempts to minimizes overall amplitude of added noise through reducing $L_2$ norm of noise.
  - constraints in the $cal(H)(T_H)$ equation above makes sure that noise gets constrained in the masked regions, sandwiched between the music amplitude and the threshold for hearing.

Use Projected Gradient Descent (PGD) as first-order optimization method to solve constrained inner minimization problem.
- Apply Stochastic Gradient Descent (SGD) for outer minimizastion

Window based Strategy for Temporal Dynamics \
- Divide the music bar $x$ into short non-overlapping windows during optimization. Each has length $l_w$.
  - for each window $x_t$, find frequency $v^m_t$ of dominant musical note for every track (track $m$) based on cumulative musical note velocities, set defensive noise $delta^m_t$ for window $t$ and track $m$ to same frequency $v^m_t$.
  - if there is no dominant musical note, no noise will be introduced. Noise is introduced uniquely for each instrumental track, so the strategy hinges upon creating defensive noises for each track.

Apply to windows using a `sigmoid()` function to constrain $delta^m_t$ to the following range for all $t$ windows $m$ tracks:
$
  cal(H)(T_H(v^m_t)) <= delta^m_t <= cal(M)(x^m_t), forall t, m
$
- $v^m_t$ is dominant frequency, $cal(M)(dot)$ is velocity of dominant musical note at bar $x^m_t$

*Black-Box Setting*
- Defender does not know which generative model is used to train music data, nor access to model for generating music
  - to make error-minimizing noise applicable to generative models, the cross-model transferability needs to be improved

Meta-Learning: strategy for tackling new tasks through _learning to learn_.
- Model first learns knowledge and finds associations / mappings amongst multiple training tasks during the meta-training phase.
- then adapts to unseen task with few examples through fine tuning (meta-testing phase)

Two-step iterative method: to generalize Unlearable Examples:
- Randomly sample $S_1, ... S_Q$ from a collection of generative models
- meta-testing and meta training

Meta-Training:\
- Segment music into windows, initialize defensive noise $delta^m_t$ for each window $t$ and track $m$
- Use first $Q - 1$ models to simulate white-box scenario to generate unlearnable music.

Bi-level optimization becomes the following:
$
  min_theta EE [min_delta (sum_(q = 1)^(Q - 1) cal(L)^q_(g e n)(f(x + delta})] + alpha sum_(m = 1)^M w^m || alpha^m||_2 \
  "s.t" \
  cal(H)(T_H) <= alpha^m <= x^m forall m in {1, 2, ... M}
$
- for $q in S$
- multiple gradients from selected models means we take average of gradients by dividing it by total number of models to ensure balanced optimization. This gives us

$
  gradient_x cal(L)_(g e n)(f(x + delta)) = 1/Q sum_(q = 1)^(Q - 1) cal(L)_(g e n)^q (f(x + delta))
$

Meta-Testing \
- done to refine noise for adapting to unseen model.
- fine-tune defensive noise for last sampled model $S_Q$ for generalizing by placing it in a black-box setting
- Divergence Loss (wtf is this) quantifies disparity between distribution of target model's output and distribution of clean music
  - simply put, how effective we are minimizing learning for the model.

We craft defensive noise with optimization objectives
- iteratively adjust perturbation (deviations) to minimize model's loss while maintaining imperceptibility for humans.
- iterative optimization process + divergence loss in black-box setting refines defensive noise

*HarmonyCloak for Wave Audio*
- Previously were talking about how to use it with MIDI, but we can adapt the system for wave-based formats.

Window the audio file, lengths of roughly 10ms, lengths of roughly 10ms. Compute the STFT to obtain dominant frequencies $(T_H(v_t))$ in dB SPL. COmpute magnitude $M(v_t)$ by computing the FFT and calculating the relative dB-SPL with
$
  MM_t = -20 dot log((M(v_t)) / (20^(-6)))
$

introduce the noise $delta_t$ for dominant frequency within each window. Magnitude must be constrained within the following range for all windows $t$:
$
  T_H(v_t) <= delta_t <= MM_t
$
- for multi-track wave based audio, operations are applied to each track individaully.

*Evaluation* \
_Experimental Setup_

Evlauation Metrics:
+ effectiveness (training loss and perceptual quality of music produced by generative models)
+ perceptual quality (generated UEs should be enjoyable to listen to, defensive noises should have minimal impact on music quality)

Intra track and Inter track metrics for music:
- Empty Bars (EB) Ratio: percentage of empty bars in generated music
- Used Pitch Classes (UPC) per Bar: number of unique pitches within each bar of generated music [0, 12].
- Qualified Notes (QN) Ratio: Quantifies percentage of qualified notes in Generated Music
  - qualified note has a duration of at least 3 time steps. provides insight into level of fragmentation or coherence in music.
- Drum Pattern (DP): ratio of notes that conform to 8-16 beat patterns. Indicates adherence to established rhythmic patterns
- Tonal Distance (TD): tonal distance between pair of tracks. Larger TD implies waeker inter-track harmonic relations
  - helps assess level of harmonicity or dissonance, ranges from 0 to 5, 0 for harmonic, 5 for weak inter-track harmonic relation
- Frechet Audio Distance (FAD): statistics on a set of reconstructed music clips to statistics on a set of studio-recorded music
  - reference-free audio quality metric, correlates well with human perception. Low FAD is better.

TD and FAD to measure perceptual quality

Eb, UPC, QN, DP to compare real music with generated samples temporally (effectiveness of generator)
- similar distributions of real and generated samples $==>$ temporal domain metrics should also exhibit proximity wtf does this mean

Experimental Settings: \
- evaluated on 3 music generative models: MuseGAN, SymphonyNet, and MusicLM.
- Lakh MIDI Dataset to train generative models and generate unlearnable examples.
- noise is introduced to only 15% of training dataset by default.

*Results:*
- Effectiveness evaluated by Training Loss Comparison, Temporal Analysis, Perceptual Analysis
- Training Loss Comparison: training loss on unlearnable music quickly approaches zero after few iterations in both white and black box settings.
- Temporal Analysis: Models trained on unlearnable music is significantly lower percentage of EB than models on clean music and even training data
  - lack of rhythm and structure as a result of training on UEs.
  - higher UPC for models with UEs.
    - music is implausible for human listeners.
  - QN percentage differs per model: notes are either shorter or longer compared to clean musical notes.
- Perceptual Analysis: All models exhibit lower TD with clean music. All models show higher TD values when trained with UEs.

Perceptual Quality \
- unlearnable music from black-box has higher TD than that in white-box (lack of direct knowledge in black box setting)
- FAD scores are similar in quality for both UE and clean music (little perceptual impact)

REsilience against Lossy Compression \
- Noise generated under $L_p$ norm constraints overwhelm original music's pitch and harmonics, completely masking them out entirely.
- also gets eliminated by MP3 compression.
- #smallcaps("HarmonyCloak") generates noise that harmonizes with music frequencies, making them less perceptible. Minimize masking-to-noise ration (MNR) so even after compression, noise maintains characteristcs and preerves unlearnabililty.

Unlearnability on Genre Classification
- Generated UEs in Rock, included it with learnable classical, country, jazz, and rock samples. Music genre classifier attempts to predict genre of current song.
- struggles identifying rock.
- results indicate that generative models have limited knowledge from rock and are trained on other genres, but this doesn't hinder model's to learn other genres.

Impact of Window Size:
- Shorter window sizes result in steeper loss curves (effectively compresses knowledge that can be learned during training)
  - shorter window sizes lead to higher FAD scores (low music quality).
- impat of window size on FAD is minimal.
- increased noise levels with short window sizes, quality of clean music is relatively the same as that of the UEs.

Impact of Unlearnable Percentages \
- Do the UEs affect learning process for other clean samples with similar features?
experiment where varying percentages of rock music were unlearnable, trained MuseGAN with partially unlearnable $cal(D_u)$ and partially clean $cal(D_c)$ (both rock) and clean music from other genres
- Rock classification declines when 80% of examples are unlearnable (remaining 20% of clean examples is not enough data)
- sample set with 20% learnable rock music is not enough data for effectivel learning.

- 80% unlearnable pop music reduces genre classifier down to 57.8%
- 75.1% when only 20% is unlearnable
  - jazz drops to 65.7%, 59% for country, 47% for classical.

they also placed UEs randomly in their training set for MuseGAN just to see if it really stops learning in its tracks.
- it turns out that overall learning does decline when analyzing EB, UPC< QN, and DP on generated music.
- classification models that primarily focus on distinguishing categories. Generative models capture underlying data distribution and learn patterns, structures, and relatinoships wthin training data for new, smimilar outputs.
  - reduction in proportion of learnable music compromises capacity to generate plausible music

*User Study*
- listening test with 31 participants (21 male, 10 female) age 25-36, self-identified music lovers.
- four sets of music clips in randomized order to eliminate bias. Each clip has 30s minimum duration, first set contains clean and UE (black and white)
- other sets contained foru music samples: 2 from MuseGAN, SymphonyNet, MusicLM on clean music, two from one of those models on unlearnable music
- rate music based on 5 point likert scale regarding
+ pleasant harmony $(H arrow.t)$
+ plausibility $(P arrow.t)$
+ presence of noise $(N arrow.t)$
+ overall rating $(O R arrow.t)$
- UEs had a drop in quality with OR scores as low as 2.22 in white-box settings

*Robustness of #smallcaps("HarmonyCloak")*
- used HarmonyCloak with Spectral Subtraction and Non-negative Matrix Factorization, two popular noise reductiont echniques as well as two defenses (EPIc and DP-InstaHide)
- SS estimates and subtracts noise spectrum from observed singal
- NMF is a matrix decomposition for audio source separation and noise reduction
  - decomposing observed signal into constituent parts lets you isolate noise components and reconstruct cleaner version
- EPIc detects and eliminates poisoned data
- DP-InstaHide augments data to eliminate data poisoning

- model trained on clean music is the best
- noise removal techniques and adaptive defenses still struggles to produce high-quality music even after using them.

*Related Work*

_Generative AI in Music_
- ILLIA Suite by Hiller
- AI based music models with neural networks emerged in 1980s with Nierhaus
- Roberts et al. proposed MuseVAE (hierarchical decode for variational autoencoder (VAE)) to address challenges of modeling long-term structure in sequential data
- Dhariwal et al. proposed Jukebox, multiscale VQ-VAWE to compress long-context raw audio and uses Autoregressive Transformers to generate high-fidelity songs with music
- MusicGAN - first multi-track generative model
- SymphonyNet (Liu et al.) permutation invariant language model for symphonic music using modified Byte Pair Encoding algo
- Mubert uses a transformer to embed text prompts and select music tags related to encoded prompt to query song generation API
- Riffuson is stable diffusion model on mel spectrograms of music pieces from msuic-text dataset (generating music based on textual imput)
- Agostinelli et al. proposed MusicLM

_Unauthorized Data Usage Prevention in AI_
- Fowl et al. conducted studies showing efficay of adversarial examples in data poisoning. he surpassed previous poisoning methods concerning secure datasets.
- Yu et al. explores indiscriminate poisoning attacks to prevent unauthorized data exploitation
  - examining linear separability of sophisticated attack perturbations
- Preventing third-party training on data without permission: Shan et al. propose privacy protection method using adversarial attacks to add perturbations that are inperceptible to users' data. Model trained on dataset are invalid; protects against unauthorized deep learning models
- Huang et al. and Fu et al. suggest unlearnable strategies using error-minimizing noise to reduce error of training.
- Liu et al. improved robustness of unlearnable examples throuhg using data's grayscale knowledge
- Ren et al. uses Classwise Separability Discriminant (CSD) to enhance transferability of UEs across different settings and datasets
- Zhao et al. - unlearnable examples for diffusion models using error-minimizing noise strategies
- Zhang et al. - label-agnostic unlearnable examples with cluster-wise perturbations
- Liu et al. - Stable unlearnable examples by training defensive noise against Random perturbation instead of adversarial perturbation for defensive noise stability
- Ye et al. proposes ungeneralizable examples which are trained from maximizing designated distance loss in common space alongside undistillation optimization (wtf are these words)

Other protective techniques:
- EditShield: method introducing distortions to protect against unauthorized image editing through misleading instruction-guided diffusion
- Liu et al. MetaCloak uses meta-learning to generate transformation-resistnat perturbations to protect personal data from text-to-image synthesis
- Shan et al. - Glaze, protect artists from style mimicry from text-to-image diffusion models
  - introduce perturbations maximizing feature differences from original image
- Shan et al. - Nightshade - prompt-specific poisoning attack for diffusion-based text-to-image models for concept destabilization
- Yu et al. AntiFake: adversarial audio system to thwart unauthorized speech synthesis; safeguards integrity of audio content from exploitative AI technologies.

*Discussions and Future Work*

- HarmonyCloak is currently focused on instrumental music, wants to extend to vocal-instrumental compositions
- Professional Musicians for Deeper Insights: include professionals across various genres and roles for nuanced and expert feedback
- Broadening Testing ot Multiple compression Formats and platforms: Streaming platforms like YouTube, Spotify, and SoundCloud have their own unique compression algorithms and bitrate settings, could interact weirdly with HarmonyCloak.
- Ensuring Long-Term Effectiveness against Evolving AI Tech: Future work is focused on strengthening robustness of perturbation-based UEs in music, drawing from lessons in the image domain.
  - ensures sustained protection of musician's rights and creative works against unauthorized exploitation through sophisticated AI Technologies
