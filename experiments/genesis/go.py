import argparse
import subprocess
import sys
import os

# Registry of all replication phases for the Genesis-152M interpretability research
# Format: 'phase_id': {
#    'name': '...', 
#    'script': '...', 
#    'args': '...', 
#    'description': '...', 
#    'narrative': '...', 
#    'technical': '...',
#    'dependencies': [...]
# }

PHASES = {
    # Earlier Phases
    "0A": {
        "name": "Corrected Effective Rank Profiling",
        "script": "scripts/run_corrected_er.py",
        "args": "--max-prompts 60 --max-tokens 64",
        "description": "Computes corrected macroscopic effective rank (ER) across depths, using Welford covariance.",
        "why": "Establishes the true spatial volume of the residual stream without short-context sequence length artifacts.",
        "narrative": "Prior metrics suggested the model utilized only 8% of its dimensions. This test proves that was a measurement error caused by short sequences. By looking at thousands of tokens across many prompts, we find the model actually uses about 32%.",
        "technical": "Shannon effective rank (Roy & Bhattacharyya 2007) is defined as the entropy of the normalized singular value distribution. When sequence length T < hidden dimension d, ER is strictly capped at T. Welford's algorithm allows accumulating the full dxd covariance matrix over N >> d samples, revealing the true heavy-tailed power-law decay. Note: MP bulk distortion occurs when N/d < 10.",
        "dependencies": []
    },
    "0B": {
        "name": "Sub-block Rank Localization",
        "script": "scripts/run_subblock_rank.py",
        "args": "",
        "description": "Measures ER at block_input, post_norm, post_mixer, and post_ffn.",
        "why": "Identifies which architectural sub-components (norm, mixer, FFN) are responsible for expanding or crushing representations.",
        "narrative": "Every layer in the model performs a 'dance' of rank: the Norm inflates it, the Mixer (Attention) crushes it, and the FFN restores it. This test pinpoints exactly where represented concepts are being squeezed.",
        "technical": "Direct sub-block kinematic profiling confirms a period-4 sub-block oscillation. ZeroCenteredRMSNorm inflates rank (+154 ER @ L15) by dampening the dominant principal component and redistributing energy to the tail. The Mixer acts as an active spectral gatekeeper, severely crushing rank (-136 ER).",
        "dependencies": []
    },
    "1A": {
        "name": "FoX Head Causal Ablation",
        "script": "scripts/run_head_rank.py",
        "args": "",
        "description": "Zero-ablates individual attention heads and measures impact on mixer ER.",
        "why": "Identifies asymmetric specialization, revealing that specific heads (e.g. L15-H3) act as massive rank compressors.",
        "narrative": "Not all 'brains' in the model do the same thing. Some attention heads are 'specialists' that aggressively filter information. By turning them off one by one, Analysis revealed the 'load-bearing' heads that hold the representation together.",
        "technical": "Ablation of L15-H3 causes mixer ER to skyrocket (+56 ER), proving this head is responsible for extreme dimensional collapse. This reveals that MHSA doesn't compress uniformly; it delegates bottlenecking to specialized circuits.",
        "dependencies": []
    },
    "2A": {
        "name": "W_o Compression Probe",
        "script": "scripts/run_wo_probe.py",
        "args": "",
        "description": "Measures ER pre-W_o (concatenated heads) vs post-W_o.",
        "why": "Demonstrates that the W_o projection matrix is an active, anisotropic funnel that crushes variance.",
        "narrative": "The model's 'attention' produces a lot of raw data, but it needs to fit back into the main 'communication bus'. Analysis proves that the output gate (W_o) acts like a funnel, shearing away unnecessary noise to keep things aligned.",
        "technical": "Rank reduction at W_o is driven by structural anisotropy, not GQA ceilings. Decomposing variance along singular directions of the output O=ZW_o shows that the spectral tail decay is dictated almost entirely by the W_o contribution (||v||^2), acting as a spectral gatekeeper.",
        "dependencies": []
    },
    "3A": {
        "name": "Prompt Types / Complexity Divergence",
        "script": "scripts/run_prompt_types.py",
        "args": "",
        "description": "Computes structural ER differences across different cognitive categories.",
        "why": "Evaluates the prior hypothesis about domain-specific volume requirements (later refined by bootstrap CI).",
        "narrative": "Do harder tasks like math require more 'mental space' than simple creative writing? Initially, hypotheses suggested so. This test compares the geometric footprint of different tasks.",
        "technical": "While initial runs suggested Math > Creative volume, Phase 7C-CI bootstrapping (1000 iterations) showed the volume gap is not statistically significant (p=0.929). However, Finding 12 (Principal Angles) confirms that while volumes are similar, the *orientations* are strictly orthogonal, indicating shared infrastructure with domain-specific branches.",
        "dependencies": []
    },

    # Phase 9: Causal Reliability Program
    "9A": {
        "name": "Semantic Extraction",
        "script": "scripts/run_phase9_extract.py",
        "args": "--max-prompts 120 --max-tokens 64",
        "description": "Extracts baseline residual stream hidden states across the prompt benchmark.",
        "why": "Generates background covariance matrices and intermediate states for downstream interventions.",
        "narrative": "Before steering the model, this phase attempts to map the 'neighborhood' of its thoughts. This phase records how the model naturally responds to 120 different prompts to create a baseline map.",
        "technical": "Global ER captures the geometric union of disjoint task manifolds. While individual prompts occupy highly collapsed fractions (ER ~4-36), the aggregated covariance establishes the 185-dim manifold ceiling. Scaling to N=3840 ensures the MP noise bulk doesn't consume signal dimensions.",
        "dependencies": []
    },
    "9B": {
        "name": "Manifold Mapping",
        "script": "scripts/run_phase9_manifold.py",
        "args": "",
        "description": "Generates manifold plots and geometry summaries.",
        "why": "Provides descriptive visualizations of the semantic state space.",
        "narrative": "This phase creates the actual maps—turning complex numbers into pictures that show how 'Math' thoughts are grouped differently from 'Story' thoughts.",
        "technical": "Visualizes the U-shaped spectral trajectory. Analysis proves that prompt diversity injects significantly more effective rank than autoregressive sequence depth, shaping the macroscopic manifold.",
        "dependencies": ["9A"]
    },
    "9C": {
        "name": "Semantic Direction Isolation",
        "script": "scripts/run_phase9_semantic_dirs.py",
        "args": "--k-bulk 70 --min-retained-fraction 0.25",
        "description": "Extracts raw and bulk-orthogonalized semantic task directions.",
        "why": "Provides the causal vectors used to steer the model's output.",
        "narrative": "The goal is to isolate the exact 'direction' in the model's mind that means 'Mathematics'. Analysis revealed the difference between Math and Creative thoughts, then cleaning up the signal.",
        "technical": "Uses difference-in-centroids (delta). To isolate the 'quiet subspace' (Finding 11), we orthogonalize delta against the top k=70 bulk eigenvectors. This reduces delta's norm by ~72x, proving semantic differences live in tiny, suppressed trailing dimensions.",
        "dependencies": ["9A"]
    },
    "9H": {
        "name": "Direction Stability Diagnostic",
        "script": "scripts/run_phase9_direction_stability.py",
        "args": "--k-bulk 70 --resamples 10",
        "description": "Tests numerical stability of extracted directions.",
        "why": "Ensures causal vectors are robust to specific prompt samples.",
        "narrative": "This test evaluates if the 'Math direction' is real, or just a fluke of the specific prompts used. This test re-runs the extraction many times with different subsets to make sure the direction is stable.",
        "technical": "Measures cosine stability across bootstrap resamples. delta_raw is typically highly stable (cos > 0.99), while delta_perp (orthogonalized) is more sensitive but stabilizes at mid-network layers like L15.",
        "dependencies": ["9A"]
    },
    "9D": {
        "name": "Layerwise Semantic Evolution",
        "script": "scripts/run_phase9_layerwise_semantics.py",
        "args": "",
        "description": "Tracks state projection onto semantic axes across layers.",
        "why": "Shows where the model structurally separates different domains.",
        "narrative": "This phase tracks the 'Math' signal as it travels through the model's 30 layers. Research indicates it starts fuzzy and becomes crystal clear right before the final answer.",
        "technical": "Finds the 'Lexical Crossover' (Finding 13). At L15, the vocabulary distribution collapses from >100 words to ~28, and domains swap confidence. This geometric bottleneck is the causal mechanism for finalizing sequence prediction.",
        "dependencies": ["9A", "9C"]
    },
    "9E": {
        "name": "Semantic Rotational Dynamics",
        "script": "scripts/run_phase9_semantic_rotation.py",
        "args": "",
        "description": "Evaluates residual stream rotation.",
        "why": "Determines if domains use static vectors or complex rotational orbits.",
        "narrative": "The model's thoughts aren't static—they spin. This phase proves that 97% of the model's internal processing is 'rotational', like a giant planetary system of features.",
        "technical": "Estimates the linear layer-to-layer operator A (h_{l+1} \\approx Ah_l). Finding 15 shows 560/576 eigenvalues are complex conjugate pairs (97.2%), falsifying the 'monotonic refinement' hypothesis in favor of a rotational linear dynamical system.",
        "dependencies": ["9A", "9C"]
    },
    "9F": {
        "name": "Causal Steering Quantification",
        "script": "scripts/run_phase9_semantic_steering.py",
        "args": "--lambda-sweep 0.0,5.0,12.5 --vector-key delta_perp --eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Steers generation using the orthogonal semantic vector.",
        "why": "Provides behavioral proof of the isolated causal direction.",
        "narrative": "The 'Moment of Truth'. Steering the model's thoughts in the 'Math' direction determines if it actually starts acting more analytical. Research indicates we can precisely nudge its behavior.",
        "technical": "Causal Steering (Finding 11) using a 3-tier Lambda sweep. At \\\\lambda=5.0, sentence length drops and geometric proximity to the Math centroid increases by 40% without destroying information entropy. \\\\lambda=12.5 causes 'coherence destruction' (ER collapse).",
        "dependencies": ["9C"]
    },
    "9G": {
        "name": "Activation Patching / Swap",
        "script": "scripts/run_phase9_activation_patching.py",
        "args": "--layer 15 --alpha-sweep 0.0,0.5,1.0 --eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Clean-to-corrupt residual stream state patching.",
        "why": "Mechanistic localization of semantic decisions.",
        "narrative": "This phase attempts to 'teleport' a specific thought from one run to another. This phase attempts to swap the model's internal activations to see if we can force a correct answer on a failing run.",
        "technical": "Standard activation patching often fails in Genesis due to high-dimensional non-linearity. While steering (adding a vector) works, patching (overwriting) is more fragile, indicating the model uses 'aligned injection' where new data must constructively interfere with existing state.",
        "dependencies": []
    },
    "9I": {
        "name": "Causal Triangle Diagnostic",
        "script": "scripts/run_phase9_causal_triangle.py",
        "args": "--layers 13,14,15,16,17 --eval-json prompts/phase9_shared_eval_heldout.json --semantic-directions logs/phase9/semantic_directions.json",
        "description": "Tests alignment between steering, patching, and head-ablation.",
        "why": "Verifies if methods converge on the same causal substrate.",
        "narrative": "This test evaluates if three different ways of poking the model all agree on where the 'Math' center is. If they all point to the same spot, Analysis revealed something real.",
        "technical": "Checks alignment between mediation (patching) and intervention (steering). Results stay inconsistent across L13-17, suggesting the causal signal is not a single persistent vector but a transformed dynamical trajectory.",
        "dependencies": ["9C"]
    },
    "9J": {
        "name": "Energy Landscape",
        "script": "scripts/run_phase9_energy_landscape.py",
        "args": "--layers 15 --eval-json prompts/phase9_shared_eval_heldout.json --semantic-directions logs/phase9/semantic_directions.json",
        "description": "Measures energy shifts under semantic perturbations.",
        "why": "Checks if interventions push states into OOD regimes.",
        "narrative": "Is steering 'natural', or is it breaking the model's brain? This test measures the 'stress' on the model's internal state during steering.",
        "technical": "Evaluates whether perturbations push the state into high-norm, off-manifold regions. Helps distinguish between 'manifold-nudges' (natural shifts) and 'manifold-shock' (destructive OOD artifacts).",
        "dependencies": ["9C"]
    },
    "9K": {
        "name": "Counterfactual Trajectory Patching",
        "script": "scripts/run_phase9_counterfactual_trajectory.py",
        "args": "--layers 13,14,15,16,17 --start-layer 15 --eval-json prompts/phase9_shared_eval_heldout.json --semantic-directions logs/phase9/semantic_directions.json",
        "description": "Multi-layer patching along forward trajectory.",
        "why": "Tests if causal states require multi-layer intervention.",
        "narrative": "This phase attempts to keep the 'Math' thought active across multiple layers to see if it can bypass the model's self-correction.",
        "technical": "Tests for 'Generative Perseveration'. If single-layer patching fails but multi-layer succeeds, it indicates the model possesses 'error-correcting' dynamics that dampen transient residual perturbations.",
        "dependencies": ["9C"]
    },
    "9L": {
        "name": "Closed-Loop Control",
        "script": "scripts/run_phase9_closed_loop_control.py",
        "args": "--layer 15 --target-levels=-3,-1,0,1,3 --eval-json prompts/phase9_shared_eval_heldout.json --semantic-directions logs/phase9/semantic_directions.json",
        "description": "Applies a PID-like semantic controller.",
        "why": "Evaluates sustained causal steering over multi-token outputs.",
        "narrative": "This phase constantly adjusts the steering to keep the model exactly as 'Analytical' as desired, token after token.",
        "technical": "Implements a feedback loop where steering magnitude \\\\lambda is adjusted based on the current state's projection onto the semantic axis. Tests the limits of 'Aligned Injection' over long sequences.",
        "dependencies": ["9C"]
    },
    "9M": {
        "name": "Token Window Patching",
        "script": "scripts/run_phase9_token_window_patching.py",
        "args": "--layers 13,14,15,16,17 --window-sizes 1,2,3,4 --eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Patches contiguous windows of token positions.",
        "why": "Tests if temporal smearing resolves patching failure.",
        "narrative": "Instead of patching one token, this phase attempts to patch a whole 'phrase' of internal states. This helps determine if concepts are spread out over time.",
        "technical": "Checks for temporal 'Innovation Ratios'. Finding 10 shows the GLA memory locks in at t=27. Window patching evaluates if concepts are accumulated or if they are transient 'impulses'.",
        "dependencies": []
    },
    "9N": {
        "name": "Directional Patching",
        "script": "scripts/run_phase9_directional_patching.py",
        "args": "--layers 15,16,17 --alpha-sweep 0.5,1.0,2.0 --eval-json prompts/phase9_shared_eval_heldout.json --semantic-directions logs/phase9/semantic_directions.json",
        "description": "Splits patching into full, semantic-only, and orthogonal components.",
        "why": "Narrows causal search to specific task subspaces.",
        "narrative": " টেলিportation attempts are broken into two parts: the Math part and 'everything else'. Analysis revealed that the Math part is indeed the key to changing the answer.",
        "technical": "Decomposes the clean-corrupt delta \\\\Delta = \\\\delta_{parallel} + \\\\delta_{perp}. Finding 11 shows semantic differences live in the 'quiet subspace'. This test isolates that subspace's contribution to the patch effect.",
        "dependencies": ["9C"]
    },
    "9O": {
        "name": "Sign-Aware Readout Refinement",
        "script": "scripts/run_phase9_semantic_readout_refinement.py",
        "args": "--layer 17 --alpha-sweep 0.5,1.0,2.0 --eval-json prompts/phase9_shared_eval_heldout.json --semantic-directions logs/phase9/semantic_directions.json",
        "description": "Evaluates patch success using strict sign-aware readouts.",
        "why": "Prevents noisy symmetric variance from spoofing results.",
        "narrative": "This phase evaluates if the model moves in the correct direction. This is a much stricter test that removes lucky guesses.",
        "technical": "Sign-consistency tests (e.g. Wilcoxon signed-rank) are robust to heavy-tailed outliers. An aggregate margin might look positive while failing per-item. 9O proves the 9N 'bump' was mostly a mixed-sign averaging artifact.",
        "dependencies": ["9C"]
    },
    "9P": {
        "name": "Readout-Localized Patching",
        "script": "scripts/run_phase9_readout_localized_patching.py",
        "args": "--layer 17 --alpha-sweep 0.5,1.0,2.0 --eval-json prompts/phase9_shared_eval_heldout.json --semantic-directions logs/phase9/semantic_directions.json",
        "description": "Isolates patch strictly to un-embedding-aligned components.",
        "why": "Tests if signal is constrained to final output axes.",
        "narrative": "Only the parts of the thought used to pick words are 'teleported'. This determines if the 'Math' concept is already being turned into speech at this layer.",
        "technical": "Splits the semantic patch into 'Readout Aligned' (projected to W_U) vs 'Orthogonal'. No rescue found here, proving the causal signal is not yet fully transformed into a simple logit bias at L17.",
        "dependencies": ["9C"]
    },
    "9Q": {
        "name": "Recurrent State Patching (9Q/9R)",
        "script": "scripts/run_phase9_recurrent_state_patching.py",
        "args": "--layers 13,14,16,17 --eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Tests if causal signal lives in GLA recurrent state.",
        "why": "Mechanistic check of state dependence via incremental forward.",
        "narrative": "Genesis has a 'long-term memory' (recurrent state). This phase attempts to teleport *that* memory instead of just the immediate thought. It works better than residual patching, but only in certain layers.",
        "technical": "GLA segment state S_t is the persistent causal variable. Finding 19 shows S_t compresses to ~13 dims. Patching S_t directly bypassing the residual add avoids 'aligned injection' dampenings, but remains weak overall.",
        "dependencies": []
    },
    "9S": {
        "name": "FoX Query Patching",
        "script": "scripts/run_phase9_fox_query_patching.py",
        "args": "--layer 15 --eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Directly patches FoX query path (q_proj).",
        "why": "Tests if useful signal is queried rather than purely residual.",
        "narrative": "This phase attempts to change what the model is 'looking for' (the Query) to see if that triggers the behavior.",
        "technical": "In MHSA, read-path depends on q = q_proj(x). Patching q_proj tests if the mechanism is 'query-conditioned'. Results were flat-to-negative, and random queries performed better, suggesting query-patching is not the bottleneck.",
        "dependencies": []
    },
    "9T": {
        "name": "Bilinear State-Query Attribution",
        "script": "scripts/run_phase9_bilinear_attribution.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Measures interaction between L14 state and L15 query.",
        "why": "Tests for joint bilinear effect neither achieved alone.",
        "narrative": "This test evaluates if both the right 'Memory' AND the right 'Search Query' are required simultaneously to unlock the correct answer.",
        "technical": "Interaction = score(both) - score(state) - score(query) + score(base). Tests for non-linear, multi-feature dependence (Boolean gating). No clean bilinear rescue was found on the 48-item benchmark.",
        "dependencies": []
    },
    "9U": {
        "name": "Token-Position Steering Sweep",
        "script": "scripts/run_phase9_token_position_steering.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json --semantic-directions logs/phase9/semantic_directions.json",
        "description": "Injects steering at early vs late positions.",
        "why": "Tests if intervention acts as trajectory selector.",
        "narrative": "Surprisingly, the model responds best to 'last-minute' steering.",
        "technical": "Falsifies the 'dynamical eigen-direction' theory where early amplification dominates. The effect is answer-adjacent, indicating that steering acts more as a late alignment operator than an attractor-based trajectory selector.",
        "dependencies": ["9C"]
    },

    # Phase 10: Mediator Discovery and Specificity
    "10A": {
        "name": "Mediator Depth Sweep",
        "script": "scripts/run_phase10_layer_depth_steering.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Reuses L15 semantic vector at different target layers.",
        "why": "Reveals FoX L11/L7 are actually stronger steering sites.",
        "narrative": "Analysis revealed that the 'Math' signal from Layer 15 actually works *better* if injected earlier, in Layer 11 or 7. This reveals a 'sweet spot' for controlling the model.",
        "technical": "The strongest shared-benchmark effect is at FoX L11 (+0.0284 margin). This points to a 'mediator corridor' where the model is most sensitive to semantic re-orientation.",
        "dependencies": ["9C"]
    },
    "10B": {
        "name": "L15-H3 Ablation Interaction",
        "script": "scripts/run_phase10_h3_ablation_interaction.py",
        "args": "",
        "description": "Checks if L15-H3 ablation blocks steering.",
        "why": "Identifies L15-H3 as participant but not sole bottleneck.",
        "narrative": "Turning off the 'Math specialist' head (H3) only partially attenuates steering. This means H3 is an important part of the path, but not the only one.",
        "technical": "L15-H3 ablation only partially attenuates L11 steering. This indicates that the mediator corridor uses multiple parallel pathways, confirming architectural redundancy.",
        "dependencies": ["9C"]
    },
    "10C": {
        "name": "OOD Transfer Sweep (Family 1)",
        "script": "scripts/run_phase10_layer_depth_steering.py",
        "args": "--eval-json prompts/phase10_ood_semantic_eval.json",
        "description": "Re-runs depth steering on first OOD family.",
        "why": "Evaluates robustness of semantic axis (reveals family-sensitivity).",
        "narrative": "This phase evaluates if the control is specific to the types of prompts initially studied. It fails here.",
        "technical": "Transfer is 'family-sensitive'. L11 steering turns negative on OOD Family 1 (-0.0151). This weakens the 'robust prompt-invariant feature' story in favor of a 'geometric alignment operator' account.",
        "dependencies": ["9C"]
    },
    "10C_B": {
        "name": "OOD Transfer Sweep (Family 2)",
        "script": "scripts/run_phase10_layer_depth_steering.py",
        "args": "--eval-json prompts/phase10_ood_semantic_eval_family2.json",
        "description": "Re-runs depth steering on second OOD family.",
        "why": "Cross-verifies family-sensitivity across a broader OOD pool.",
        "narrative": "A second set of new prompts shows a weak positive effect. This confirms the model's 'Math' neighborhood is complex.",
        "technical": "OOD Family 2 shows a weak positive effect (+0.0088). This confirms the steering direction is meaningful (beats random) but its effectiveness is highly dependent on the local 'decision geometry'.",
        "dependencies": ["9C"]
    },
    "10D": {
        "name": "Orthogonal Residual Decomposition",
        "script": "scripts/run_phase10_residual_decomposition.py",
        "args": "",
        "description": "Compares axis_ablate vs add vs axis_ablate_plus_add.",
        "why": "Proves mechanism acts as alignment operator, not static feature.",
        "narrative": "Analysis revealed the 'Math' direction is mostly empty (0.2%) in the model's natural state, meaning a new nudge is being created rather than changing a stored thought.",
        "technical": "Ablating the semantic axis causes zero degradation (energy fraction < 1%). This is uninformative regarding functional role (Cluster 3). However, 'ablate-then-add' is stronger than 'add', indicating we are overcoming internal interference suppression.",
        "dependencies": ["9C"]
    },
    "10E": {
        "name": "L11 Subcomponent Localization",
        "script": "scripts/run_phase10_l11_subcomponent_steering.py",
        "args": "",
        "description": "Pinpoints intervention within L11 sub-modules.",
        "why": "Finds attention-output corridor is the strongest handle.",
        "narrative": "Analysis revealed the 'Attention Output' is the perfect spot for intervention.",
        "technical": "The strongest Handle is the attention-output corridor (attn_output / o_proj). Early handles like q_proj or v_proj are flat-to-negative, confirming it's a late mediator / output-path alignment effect.",
        "dependencies": ["9C"]
    },
    "10F": {
        "name": "L11 Persistence and Readout",
        "script": "scripts/run_phase10_l11_persistence_readout.py",
        "args": "",
        "description": "Tracks downstream state changes after L11 injection.",
        "why": "Checks if semantic vector persists to output layer.",
        "narrative": "Interventions get 'translated' and changed as they move through the model.",
        "technical": "Persistence cosine at L29 is only 0.031. The effect survives but the vector is transformed. This fits the 'alignment operator' account better than a 'static feature' account.",
        "dependencies": ["9C"]
    },
    "10G": {
        "name": "Localized OOD Replay",
        "script": "scripts/run_phase10_localized_ood_replay.py",
        "args": "",
        "description": "OOD steering specifically at narrowed L11 sites.",
        "why": "Determines if subcomponent targeting rescues OOD portability.",
        "narrative": "This test evaluates if subcomponent targeting rescues OOD portability. It does.",
        "technical": "Narrowing to localized sites (attn_output) does not rescue OOD robustness. The fragility is a property of the broader corridor, not a site-specific artifact.",
        "dependencies": ["9C"]
    },
    "10H": {
        "name": "Perturbation Boundary Sweep",
        "script": "scripts/run_phase10_perturbation_boundary.py",
        "args": "",
        "description": "Examines downstream growth around L11.",
        "why": "Reveals L11 acts as stabilization/routing point.",
        "narrative": "In Layer 11, the 'Math' push causes *less* chaos than a random push, meaning the model 'likes' the direction more.",
        "technical": "L11 shows significantly lower final-layer divergence and growth ratios for semantic vs random perturbations. This indicates L11 is a 'stabilizing / routing-sensitive' interface where perturbations are selectively integrated.",
        "dependencies": ["9C"]
    },
    "10I": {
        "name": "Descriptive Geometry Profile",
        "script": "scripts/run_phase10_geometry_profile.py",
        "args": "",
        "description": "Layerwise representation geometry profiling.",
        "why": "Shows L11 is intermediate structural transition point.",
        "narrative": "Layer 11 is a staging area—more complex than the start, but not as wild as the middle.",
        "technical": "participation ratio and feature variance show L11 is intermediate between L7 and L15. It shows a mild entropy dip, descriptive of a transition/staging point rather than a standalone manifold collapse.",
        "dependencies": []
    },
    "10J": {
        "name": "Cross-Layer FoX Site Comparison",
        "script": "scripts/run_phase10_fox_site_comparison.py",
        "args": "",
        "description": "Compares narrowed sites across L7, L11, and L15.",
        "why": "Demonstrates steering is a broader L7-L11 corridor effect.",
        "narrative": "Layers 7 and 11 act almost like twins, while Layer 15 is much weaker. A whole 'control band' has been identified.",
        "technical": "L7 and L11 are nearly identical on the held-out benchmark (+0.028 margin). L15 is 4x weaker. This favors a broader 'mediator window' theory over an L11-exclusive handle.",
        "dependencies": ["9C"]
    },
    "10K": {
        "name": "Cross-Layer Localized OOD Replay",
        "script": "scripts/run_phase10_cross_layer_ood_replay.py",
        "args": "",
        "description": "OOD tests across L7, L11, L15 corridor-candidate sites.",
        "why": "Proves OOD fragility spans entire corridor.",
        "narrative": "The 'fragility' found earlier belongs to the whole control band.",
        "technical": "Reproduces the family-sensitive failure across the full L7-L11 corridor. Neither layer provides a reviewer-clean portable control site.",
        "dependencies": ["9C"]
    },
    "10L": {
        "name": "State-Conditional Corridor Swap",
        "script": "scripts/run_phase10_state_conditional_swap.py",
        "args": "",
        "description": "Replaces L7/L11 recipient state with donor states.",
        "why": "Tests for hidden-state-geometry dependence.",
        "narrative": "This test evaluates if steering works across internal state swaps. It does, meaning the 'Math' button doesn't depend on the exact tiny details of the current thought.",
        "technical": "Norm-matched donor replacement only mildly attenuates the steering effect. Mean donor-recipient cosine is 0.97-0.99. This proves the mechanism isn't brittle to exact local state, but is only a partial discriminator for operator vs feature.",
        "dependencies": ["9C"]
    },
    "10M": {
        "name": "Low-Overlap State-Swap",
        "script": "scripts/run_phase10_low_overlap_state_swap.py",
        "args": "",
        "description": "Forces minimum-cosine donors for stricter swap test.",
        "why": "Strengthens state-conditional operator evidence at L7.",
        "narrative": "Analysis revealed steering starts to weaken when using very different prompts, especially in Layer 7. This means the model *is* paying some attention to the local context.",
        "technical": "Deliberately low-overlap donor replacement (min_cosine) produces clearer attenuation at L7. This strengthens the 'state-conditional operator' story, as incompatible structural data ('dark matter') triggers destructive interference.",
        "dependencies": ["9C"]
    },
    "10N": {
        "name": "Gain vs Baseline Projection Diagnostic",
        "script": "scripts/run_phase10_gain_vs_projection.py",
        "args": "",
        "description": "Correlates baseline occupancy of axis with per-item gain.",
        "why": "Disproves simple 1D baseline-feature mechanism.",
        "narrative": "Steering gain is independent of what the prompt looked like before intervention.",
        "technical": "Correlations between baseline projection and steering gain are near zero (L7 r=+0.02, L11 r=-0.05). Falsifies the simple story that we are just amplifying existing feature occupancy.",
        "dependencies": ["9C"]
    },
    "10O": {
        "name": "Corridor-Input Residual Subspace Diagnostic",
        "script": "scripts/run_phase10_corridor_input_subspace.py",
        "args": "",
        "description": "Tests if a low-rank block_input PCA subspace predicts gain.",
        "why": "Supports in-domain corridor-subspace dependency.",
        "narrative": "Analysis revealed a 'neighborhood' (a subspace) that matters. If a prompt enters Layer 11 in this specific neighborhood, steering works 70% better.",
        "technical": "Held-out-success PCA subspaces (rank 4-16) explain >94% variance. Occupancy of this subspace predicts steering gain (L7 r=+0.71, L11 r=+0.65). However, this relationship collapses on OOD sets, indicating the 'neighborhood' is family-specific.",
        "dependencies": ["9C"]
    },
    "10P": {
        "name": "Causal Corridor-Subspace Intervention",
        "script": "scripts/run_phase10_causal_subspace_intervention.py",
        "args": "--alpha 12.5 --eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Ablates held-out-success PCA subspace during steering.",
        "why": "Tests necessity of low-rank corridor geometry for causal effect.",
        "narrative": "Blocking the identified neighborhood weakens the steering effect in Layer 11. This proves the neighborhood is necessary.",
        "technical": "Ablating the held-out-success block_input PCA subspace attenuates the semantic-minus-random gap in L11 slices. In-sample bias is expected given N=25 (Cluster 2), but directional necessity is observed.",
        "dependencies": ["10O"]
    },
    "10Q": {
        "name": "Crossfit Subspace Necessity Diagnostic",
        "script": "scripts/run_phase10_crossfit_subspace_necessity.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Tests if cross-family subspaces can rescue OOD steering.",
        "why": "Arbitrates between family-locked and universal manifold stories.",
        "narrative": "Stricter rules to avoid fit/eval leakage make results fuzzy, meaning 'necessity' isn't simple.",
        "technical": "Uses leave-one-pair-out cross-fitting. Only L7 rank 8 remains suggestive (p=0.057). Out-of-sample occupancy crash (Cluster 2) confirms that many identified features were spurious alignments with sample noise.",
        "dependencies": ["10P"]
    },
    "10R": {
        "name": "Crossfit Subspace Overwrite/Rescue Test",
        "script": "scripts/run_phase10_crossfit_subspace_overwrite.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Tests sufficiency via cross-fit clean donor overwrite inside the subspace.",
        "why": "Checks if transplanting low-rank corridor geometry can rescue the causal steering effect.",
        "narrative": "Transplanting brain states works for some prompt families but doesn't create a 'universal rescue', proving the model is highly context-sensitive.",
        "technical": "Primary sufficiency endpoint (semantic-PCA-vs-random overwrite-rescue) is null on held-out and Family 1, borderline positive on Family 2 (+0.016, p=0.0496). Overwrite-rescue is more consistent than ablation, but remains family-specific.",
        "dependencies": ["10Q"]
    },
    "10S": {
        "name": "Within-Band GLA-vs-FoX Attribution",
        "script": "scripts/run_phase10_within_band_attribution.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Compares semantic intervention across layers 7-11.",
        "why": "Determines if the mediator effect is uniquely FoX-specific.",
        "narrative": "The 'Math' handle is a property of the whole depth-band, not just the architecture type.",
        "technical": "Grouped architecture contrast (FoX 7/11 vs GLA 8/9/10) is null (p=0.912). The active L7-L11 band is depth-dependent, not mixer-dependent. This shifts focus from 'mixer-specialization' to 'depth-localized routing horizons'.",
        "dependencies": ["9C"]
    },
    "10T": {
        "name": "Family-Specific Direction Portability",
        "script": "scripts/run_phase10_family_direction_transfer.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Tests if separate directions rescue the mixed OOD pattern.",
        "why": "Arbitrates between 'wrong vector' and 'fragile corridor' accounts.",
        "narrative": "Custom vectors for every prompt family didn't help. The model's sensitivity is deeper than just the direction used.",
        "technical": "Fitting separate semantic directions for each prompt family. Uses leave-one-pair-out cross-fitting to prevent leakage. Finding: same-family directions do not clearly rescue OOD steering, suggesting mechanism fragility at the corridor level rather than a direction-portability bug.",
        "dependencies": ["9C"]
    },
    "10U": {
        "name": "Tail-Conditioned Same-Family Necessity",
        "script": "scripts/run_phase10_tail_conditioned_necessity.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Tests necessity exclusively on high-occupancy items.",
        "why": "Checks if low natural occupancy masks a real necessity signal.",
        "narrative": "Ablation remains mostly null even in the high-occupancy tail, making it hard to say this specific direction is 'necessary'.",
        "technical": "Target-site directions fit at L7/L11 attn_output. Slices data by signed semantic-cosine tail occupancy (top q67, q80). Ablation remains mostly null even in the high-occupancy tail, indicating that unconditional low-occupancy averaging was not the primary reason for weak necessity signals.",
        "dependencies": ["9C"]
    },
    "10V": {
        "name": "Low-Occupancy Same-Family Sufficiency",
        "script": "scripts/run_phase10_low_occupancy_sufficiency.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Tests sufficiency exclusively on low-occupancy items.",
        "why": "Checks if additive steering acts as an occupancy-gated rescue.",
        "narrative": "The intervention targets the prompts with the weakest Math signal and try to boost them. It works a little bit for some families, but it's not a universal fix.",
        "technical": "Complementary to 10U. Adds same-family direction to bottom q20/q33 tail. One narrow positive matched-control slice found (Family 2, L7), but no held-out replication. Further evidence that the steering effect is family-specific rather than a robust portable feature.",
        "dependencies": ["9C"]
    },
    "10W": {
        "name": "Natural Semantic-Scalar Interchange",
        "script": "scripts/run_phase10_natural_scalar_interchange.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Transplants donor semantic scalar into recipient.",
        "why": "Tests if concepts are stored as portable 1D scalar values.",
        "narrative": "This phase tests if specific concepts are stored as simple 'values' that can be swapped between prompts. The 'strength' of a thought is transplanted from one prompt to another. Results suggest thoughts are more complex than just a single portable number.",
        "technical": "Paired opposite-label natural semantic-scalar interchange at L7/L11 attn_output. Uses leave-one-pair-out target-site directions. Results (Finding 10W) showed non-significant semantic-vs-random contrasts and reversals in high-occupancy slices, arguing against a portable 1D stored-value account of the corridor.",
        "dependencies": ["9C"]
    },
    "10X": {
        "name": "Natural Orthogonal-Remainder Interchange",
        "script": "scripts/run_phase10_natural_orthogonal_interchange.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Transplants donor orthogonal remainder while preserving recipient scalar.",
        "why": "Tests if distributed context reuse rescues concept steering.",
        "narrative": "Everything *except* the concept (the distributed 'remainder' of the thought) is swapped. This tests if the model's overall 'mindstate' can be reused to support a thought. Some hints of reuse are found, but it's not a complete 'teleportation' win.",
        "technical": "Distributed-remainder interchange preserving the recipient's semantic scalar. Evaluates absolute toward-donor shifts. Results (Finding 10X) showed modest gains at L11, but the effect remains held-out-skewed and weak in absolute terms, suggesting distributed donor-state reuse is at most a suggestive hint.",
        "dependencies": ["9C"]
    },
    "10Y": {
        "name": "Natural Paired-Donor Bundle Interchange",
        "script": "scripts/run_phase10_natural_bundle_interchange.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Transplants donor scalar and orthogonal remainder as a coordinated bundle.",
        "why": "Tests if interaction between components is necessary for concept teleportation.",
        "narrative": "What if you need the *whole* thought to make it work? Analysis revealed that coordinated bundle transplant doesn't beat component-only swaps, and adding the concept scalar back in can actually wash out the signal. Thoughts are distributed.",
        "technical": "Tests scalar-only, orthogonal-only, and full-donor bundle reuse. Results (Finding 10Y) showed coordinated bundles underperform orthogonal-remainder transplant on held-out L11. This argues against a simple interactive natural-interchange mechanism.",
        "dependencies": ["9C"]
    },
    "10Z": {
        "name": "Post-10Y Synthesis Planning Gate",
        "script": "scripts/analyze_phase10_post10y_synthesis.py",
        "args": "",
        "description": "Artifact-only synthesis pass over 10W-10Y.",
        "why": "Executes the planning gate to determine the next justified experimental move.",
        "narrative": "A look back at all the teleportation attempts. The data confirms that simple scalar swaps and full-bundle swaps failed, leaving only a small hint in the 'remainder' of the thought. This points the way to Phase 11.",
        "technical": "Artifact-only analysis. Summarizes executed 10W-10Y artifacts on primary all-items slices. Finds no moderate scalar-interference structure and no all-items absolute-positive semantic wins, favoring decomposition over further scalar-blend variants.",
        "dependencies": ["10W", "10X", "10Y"]
    },
    "11A": {
        "name": "Orthogonal-Remainder Component Decomposition",
        "script": "scripts/run_phase11_orthogonal_remainder_components.py",
        "args": "--eval-jsons prompts/phase9_shared_eval_heldout.json --dataset-labels heldout_shared",
        "description": "Decomposes the 10X hint into cross-fit basis components.",
        "why": "Tests if specific structured components of the orthogonal remainder host portable donor features.",
        "narrative": "If the whole 'remainder' showed a hint of a thought, can we find the specific parts of that remainder that matter? This phase breaks the remainder into a ladder of components to find the core signal.",
        "technical": "Scaffold for cross-fit orthogonal-remainder component decomposition. Supports cumulative-rank plus optional single-PC conditions and matched random subspace controls. Enforces strict dual-gate promotion criteria (semantic-vs-random + semantic-vs-zero).",
        "dependencies": ["10X"]
    },
    "11B": {
        "name": "Phase 11 Near-Miss Report",
        "script": "scripts/analyze_phase11_near_misses.py",
        "args": "--summary-csv logs/phase11/orthogonal_remainder_component_summary.csv",
        "description": "Artifact-only report ranking non-promoted pooled near-misses.",
        "why": "Provides transparency on the strongest candidate components even if they fail the strict promotion gate.",
        "narrative": "Even if we don't find a perfect 'teleportation' win, this report shows us the closest candidates we found.",
        "technical": "Artifact-only analysis. Rankings are based on non-promoted pooled all-items rows, summarizing fold-level sign support and site ranking.",
        "dependencies": ["11A"]
    },
    "12A": {
        "name": "Circuit-Closure Interaction",
        "script": "scripts/run_phase12_h3_steering_circuit_closure.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Tests interaction between L15-H3 ablation and L11 steering.",
        "why": "Determines if steering and ablation target the same causal circuit.",
        "narrative": "Does the model's 'Math brain' (Head 3) and its 'Math handle' (Layer 11) work together? This test evaluates if turning off one affects the other in a predictable way.",
        "technical": "Evaluates pair-level signed_label_margin interaction. Results (Finding 12A) were non-decisive, suggesting effects are largely additive rather than super-additive on this benchmark.",
        "dependencies": ["9C", "10A"]
    },
    "12B": {
        "name": "Position / Duration Sweep",
        "script": "scripts/run_phase12_position_duration_sweep.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Sweeps answer-relative offsets and window sizes for steering.",
        "why": "Pinpoints the temporal access window for the mediator corridor.",
        "narrative": "When is the best time to nudge the model? Last minute, or way ahead of time? Analysis revealed the model is most responsive right before it starts speaking (t-1).",
        "technical": "Grid of offsets 1/2/4/8 x window 1/2/4. Results (Finding 12B) showed steering is sharply answer-adjacent, supporting a late-interface account over a persistent semantic-vector story.",
        "dependencies": ["10E"]
    },
    "12C": {
        "name": "Joint L7 + L11 Synergy Test",
        "script": "scripts/run_phase12_joint_l7_l11_synergy.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Tests if simultaneous L7 and L11 steering is super-additive.",
        "why": "Checks for serial synergy vs redundancy in the mediator corridor.",
        "narrative": "If we nudge the model in two 'sweet spots' at once, do we get a double boost? Surprisingly, no. The spots are partly redundant, like two doors to the same room.",
        "technical": "Joint interaction = joint - early - late + baseline. Results (Finding 12C) were null, arguing against a clean serial two-stage handoff in favor of a broad redundant access band.",
        "dependencies": ["10J"]
    },
    "12D": {
        "name": "FoX Forget-Gate Local-Control Test",
        "script": "scripts/run_phase12_fox_forget_gate_local_control.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Directly drives the L11 FoX forget-gate path.",
        "why": "Tests if local gate control is sufficient for behavioral change.",
        "narrative": "We tried to control the 'Forget Gate'—the part of the model that decides what to ignore. We can move the gate, but it doesn't change the model's final answer. The gate is a participant, not the boss.",
        "technical": "Intervenes at fgate_proj_input. Results (Finding 12D) showed strong local gate response but zero behavioral change, proving local gate control is insufficient for semantic steering.",
        "dependencies": ["10E"]
    },
    "12E": {
        "name": "W_o Mediation Probe",
        "script": "scripts/run_phase12_wo_mediation_probe.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Tests steering within top vs tail singular subspaces of L11 W_o.",
        "why": "Isolates the role of anisotropic gain in the mediator corridor.",
        "narrative": "Analysis revealed that the model's internal 'amplifier' (W_o) has a favorite direction. If we push in that direction, even random noise works. The 'sweet spot' is partly just an easier place to be heard.",
        "technical": "Compares semantic vs random in top-8 and tail-8 singular subspaces. Results (Finding 12E) favored an anisotropic gain-stage interpretation over a semantic-specific circuit claim.",
        "dependencies": ["10E", "2A"]
    },
    "12F": {
        "name": "Sparse Head-Bundle Closure",
        "script": "scripts/run_phase12_sparse_head_bundle_closure.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Tests necessity of sparse upstream head bundles for corridor steering.",
        "why": "Evaluates distributed-input closure for the late-mediator interface.",
        "narrative": "We turned off small groups of 'brain cells' upstream to see if it blocked the 'handle' at Layer 11. It didn't. The signal is more spread out than a few specific heads.",
        "technical": "Tests preregistered head bundles from Phase 1. Results (Finding 12F) showed non-decisive bundle interactions, strengthening the non-specific access/gain account of the corridor.",
        "dependencies": ["1A", "10E"]
    },
    "13A": {
        "name": "Pair-Susceptibility Audit",
        "script": "scripts/analyze_phase13a_pair_susceptibility.py",
        "args": "--representative-csvs logs/phase12/phase12b_t_minus_1_window1_results.csv,logs/phase12/phase12d_results.csv,logs/phase12/phase12e_results.csv,logs/phase12/phase12f_results.csv --eval-json prompts/phase9_shared_eval_heldout.json",
        "description": "Artifact-only cross-family audit of pair-level susceptibility.",
        "why": "Determines if steering gain tracks item-specific geometry or global labels.",
        "narrative": "A look at individual prompts reveals that some are just easier to nudge than others, regardless of what we're testing. The 'mental geometry' of the prompt itself is the best predictor of success.",
        "technical": "Correlation audit across 4 late mechanism families. Results (Finding 13A) showed stable pair-local structure (r=0.31) strongly predicted by baseline margin (r=0.73), favoring an upstream-state access account.",
        "dependencies": ["12B", "12D", "12E", "12F"]
    },
    "13B": {
        "name": "Bounded Upstream-State Causal Test",
        "script": "scripts/run_phase13b_bounded_state_causal_test.py",
        "args": "--eval-json prompts/phase9_shared_eval_heldout.json --top-pairs heldout_pair_1,heldout_pair_10,heldout_pair_21,heldout_pair_4,heldout_pair_7",
        "description": "L14 GLA recurrent-state overwrite on high-susceptibility pairs.",
        "why": "Directly tests if the late corridor depends on upstream recurrent state.",
        "narrative": "Final test: we tried to teleport the 'long-term memory' from one prompt to another on the prompts that seemed most sensitive. It moved the model, but not in a perfectly reliable way. The mystery of the 'Math' thought remains partly distributed.",
        "technical": "L14 recurrent-state overwrite on the final-clause window. Results (Finding 13B) were mildly positive vs baseline but null vs random, failing to establish a semantic-specific causal rescue.",
        "dependencies": ["13A", "9Q"]
    }
}

def print_banner(text):
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80 + "\n")

def run_phase(phase_id):
    if phase_id not in PHASES:
        print(f"Error: Phase '{phase_id}' not found.")
        sys.exit(1)
        
    phase = PHASES[phase_id]
    
    # Check dependencies
    for dep in phase['dependencies']:
        if dep not in PHASES:
            print(f"[!] Warning: Dependency '{dep}' for phase '{phase_id}' not found in registry.")
            continue
        print(f"[*] Note: Phase {phase_id} depends on Phase {dep}. Ensure {dep} has been run.")

    print_banner(f"Running Phase {phase_id}: {phase['name']}")
    print(f"Testing: {phase['description']}")
    print(f"Narrative: {phase['narrative']}")
    print(f"Script: {phase['script']} {phase['args']}")
    print("-" * 80)
    
    cmd = f"python {phase['script']} {phase['args']}".strip()
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"\n[+] Phase {phase_id} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n[-] Phase {phase_id} failed with exit code {e.returncode}.")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="PRISM Replication CLI for Genesis-152M")
    parser.add_argument('phase', nargs='?', help="Phase ID to run (e.g., 9A, 10O, all).")
    parser.add_argument('--list', action='store_true', help="List all available phases.")
    parser.add_argument('--info', type=str, help="Show detailed information for a specific phase.")
    
    args = parser.parse_args()
    
    def phase_sort_key(pid):
        import re
        match = re.search(r'(\d+)([A-Z_0-9]*)', pid)
        if match:
            num = int(match.group(1))
            suffix = match.group(2)
            return (num, suffix)
        return (0, pid)

    if args.list:
        print_banner("Available Replication Phases")
        sorted_phases = sorted(PHASES.keys(), key=phase_sort_key)
        for pid in sorted_phases:
            pinfo = PHASES[pid]
            print(f"[{pid.ljust(5)}] {pinfo['name']}")
        print("\nRun a specific phase with: python go.py <Phase_ID>")
        sys.exit(0)
        
    if args.info:
        if args.info.upper() not in PHASES:
            print(f"Error: Phase '{args.info}' not found.")
            sys.exit(1)
        pinfo = PHASES[args.info.upper()]
        print_banner(f"Phase {args.info.upper()} Info: {pinfo['name']}")
        print(f"Simple Narrative:\n{pinfo['narrative']}\n")
        print(f"Technical Intel:\n{pinfo['technical']}\n")
        print(f"Script:       {pinfo['script']}")
        print(f"Args:         {pinfo['args']}")
        print(f"Description:  {pinfo['description']}")
        print(f"Dependencies: {', '.join(pinfo['dependencies']) if pinfo['dependencies'] else 'None'}")
        print("\nSee deeper answers at: https://notebooklm.google.com/notebook/1a68b472-4bac-4293-8a5e-04452633415b")
        sys.exit(0)

    if not args.phase:
        parser.print_help()
        sys.exit(0)

    phase_target = args.phase.upper()
    
    if phase_target == "ALL":
        sorted_phases = sorted(PHASES.keys(), key=phase_sort_key)
        for pid in sorted_phases:
            run_phase(pid)
    else:
        run_phase(phase_target)

if __name__ == "__main__":
    main()
