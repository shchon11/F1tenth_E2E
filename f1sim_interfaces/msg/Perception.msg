# What the policy node worked out about the scan it just planned from, for the controller's
# clearance arm. Split out of the plan on purpose: `Plan` is arm-agnostic ("nothing on this topic
# knows which arm the controller runs"), and this is exactly the opposite -- it exists so that one
# arm can read what one network computed.
#
# It carries only things the policy node ALREADY computes for its own observation: the front-end's
# per-beam floor class (`f1sim.learn.frontend`) and the attitude those channels were built with.
# Nothing here is a new sensor and nothing here is required: a controller that never receives this
# topic falls back to its own geometric estimate, which is what it does today.
#
# `header.stamp` is the stamp of the /scan this belongs to, exactly as `Plan`'s is, so the
# controller pairs it with the same sensor snapshot it pairs the plan with.
std_msgs/Header header

# Per-beam P(floor) in [0, 1], at the OBSERVATION's beam count, or empty when this checkpoint
# carries no front-end. Empty is not "no floor anywhere": it is "I have nothing to say", and the
# controller then uses its own geometry.
float32[] floor

# The (roll, pitch) [rad] the policy's own observation columns were built with, and whether the
# estimate is usable at all. `source` is "vesc" (the orientation quaternion), "ego"
# (`floor.EgoStateAttitude`) or "frontend" (the network's own head).
float32[2] attitude
bool attitude_ok
string source

# The nearest car the policy's own nearest-car output reads in this scan (`f1sim.learn.aux_v2.NearestCarHead`):
# [exists logit, dx / 5, dy / 2, closing / 5, v_lat / 2], relative to the LiDAR (x forward, y left; closing [m/s] along
# the line of sight, v_lat [m/s] its sideways speed relative to us). The controller's speed limit for a car ahead reads
# it (`f1sim.opponent_speed_limit`, parameter `opponent_speed_limit`). Empty when the checkpoint has no such output.
float32[] nearest_car
