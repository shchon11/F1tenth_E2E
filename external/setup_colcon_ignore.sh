#!/bin/bash
# After `git submodule update --init --recursive`: mark the third-party packages that colcon must
# not build (the sim replaces vesc_driver + urg_node; the rest are maps / references only).
cd "$(dirname "$0")"
for p in diagnostics/diagnostic_aggregator diagnostics/diagnostic_common_diagnostics diagnostics/diagnostic_remote_logging \
         diagnostics/diagnostics diagnostics/self_test f1tenth_gym f1tenth_maps f1tenth_racetracks f1tenth-racing-stack-ICRA22 \
         f1tenth_system/teleop_tools f1tenth_system/vesc/vesc_driver korea_teams/F1tenthKorea2024 \
         korea_teams/f1tenth_tada korea_teams/KORA_K3 \
         korea_teams/project_The_4th_F1TENTH_Korea_championship; do
  [ -d "$p" ] && touch "$p/COLCON_IGNORE"
done
