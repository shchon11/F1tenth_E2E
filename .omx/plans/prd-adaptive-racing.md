# Product requirements

The ordinary user selects a driving policy and scenario, not a controller experiment arm. The policy preserves D3's obstacle/opponent behavior while learning to request faster speeds when current friction and geometry permit. The controller uses causal sensors and learned friction belief, a physically reachable local speed plan and feedback tracking. The G-force panel distinguishes truth, estimated belief, applied authority, waiting and faults. Training and evaluation are reproducible and resumable; unsuccessful candidates remain unapproved.

Technical contract and phases: autopilot-spec.md and autopilot-impl.md. User authorizes training/implementation without routine approval. No real-car actuation or cloud/public publishing is included.
