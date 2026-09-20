# f1sim_interfaces

The two messages on the boundary between `policy_node` and `controller_node`:

* `Plan` — the policy's output. A header stamped with the `/scan` the plan was computed from, the
  raw normalized 8-float action, the checkpoint that produced it and a sequence number.
* `PolicyState` — what the policy knows about itself: episode memory, staleness, inhibition.

It is a `rosidl` package, so it has to be built before anything can import it:

```bash
cd ~/F1tenth                       # or any colcon workspace with this repo's packages in src/
colcon build --packages-select f1sim_interfaces f1sim_ros
source install/setup.bash
```

Without that build, `import f1sim_interfaces.msg` fails and the graph's tests skip with a message
saying so. Nothing else in `f1sim/` depends on it: the training side never sees these messages, and
`learn/bagdata.py` reads a bag that has no `/f1sim/plan` in it without complaint.

Dependencies are `std_msgs` and nothing else, deliberately — a message package that drags in a
simulator is a message package the car cannot build.
