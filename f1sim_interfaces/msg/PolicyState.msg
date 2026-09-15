# What the policy node knows about itself: whether it is producing plans, and what its episode
# memory is doing. Diagnostic only -- nothing in the control path reads it.
std_msgs/Header header

# Does this checkpoint carry episode state (a recurrent hidden state, a decayed scan-occupancy
# channel, or both)? `memory_kind` is `f1sim.learn.memory.describe`, empty for a feedforward one.
bool memory_present
string memory_kind

# Episode memory clears since start, and the monotonic time of the last one (-1.0 before the
# first). The two places that clear it are a scan stream that came back after a gap and a
# message on /f1sim/reset; `memory_cleared_reason` says which it was.
uint32 memory_clears
float64 memory_cleared_at
string memory_cleared_reason

# Is the node refusing to produce a plan right now, and what is stale?
bool inhibited
string[] stale

# Age of each input at the time this message was built [s]; -1.0 for one never seen.
float64 scan_age_s
float64 imu_age_s
float64 odom_age_s
float64 attitude_age_s

# The plan this state belongs to: the `seq` and `checkpoint` of the last plan published.
uint32 seq
string checkpoint
