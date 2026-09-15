# The policy's output, on the wire between policy_node and controller_node.
#
# `plan` is the raw normalized action the actor emitted, in [-1, 1], exactly as
# `f1sim.mpc.decode` expects it: six curvature knots then two speed targets. It is
# ARM-AGNOSTIC on purpose -- nothing on this topic knows whether the controller will run
# `legacy`, `fixed_low`, `+clearance` or the traction guard, so the same plan stream can be
# replayed through any of them and the arms stay comparable.
#
# `header.stamp` is the stamp of the /scan message this plan was computed from, not the time
# the plan was published: the controller measures the plan's age against the sensor it came
# from, and a publish-time stamp would hide the inference latency inside it.
std_msgs/Header header

# 6 curvature knots + 2 speed targets, normalized to [-1, 1]. Fixed length: an 8-float plan is
# the action space (f1sim.mpc.ACT_DIM), and a length that can vary is a length that can be wrong.
float32[8] plan

# The checkpoint the plan came from, as `<run>@<sha256[:12]>`. Carried on every message rather
# than latched once: a controller that started after the policy, or reconnected to a second one,
# still knows what it is tracking, and a bag says which weights drove the car.
string checkpoint

# Monotonic counter, one per published plan, starting at 0. A gap is a dropped plan; a repeat is
# the controller holding one, and it can tell the two apart.
uint32 seq
