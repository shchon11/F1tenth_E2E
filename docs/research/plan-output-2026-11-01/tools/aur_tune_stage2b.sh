#!/bin/bash
# 2단계 뒤 남은 것: 0.8배에서 빠른 완만한 구간은 바깥으로 0.036 m(속도와 함께 커짐: 0.6배 0.005, 1.0배 0.06), 급한 굽이는 안쪽으로 0.054 m.
# 가설: 조향을 가속 중에는 0.94배, 감속 중에는 1.198배 하는 두 값(acc/dec_scaler_for_steer; 다른 차·다른 하한에서 맞춘 값)이 원인.
#       가속 중(굽이 나와 빠른 구간) 조향이 모자라 바깥으로, 감속 중(굽이 들어갈 때) 조향이 넘쳐 안쪽으로.
# 바꾸는 것: 그 두 값만 1.0으로(하나씩, 그리고 둘 다). 나머지는 1단계에서 고른 설정(t_clip_min 0.65).
# 재는 것: 자기 선과의 옆 거리 RMSE, 빈 트랙·장애물 점 없음, 차 128대 x 25초, 같은 seed.
# 반증 조건: (1) acc 값을 1.0으로 해도 0.8배 곧은 구간의 바깥 치우침(-0.036 m)이 절반 아래로 안 준다 -> 원인 아님, 그대로 둠.
#            (2) dec 값을 1.0으로 해도 0.6·0.8배 급한 굽이의 안쪽 치우침(+0.054 m)이 절반 아래로 안 준다 -> 원인 아님, 그대로 둠.
#            (3) 바꾼 값이 0.6·0.8배 RMSE 합(0.082 m)을 줄이지 못한다 -> 채택 안 함. 1.0배 칸은 고르는 데 안 씀(타이어 한계, 재현 폭 0.012 m 넘는 악화만 탈락 사유).
S=/tmp/claude-1000/-home-shchon11-F1tenth-E2E/b681fa17-d53e-4caa-8c76-9ce9b0410ed3/scratchpad
bash $S/aur_one.sh as_1.0 A '{"follower": "map"}' '{"t_clip_min": 0.65, "acc_scaler_for_steer": 1.0}' 256 25 oracle
bash $S/aur_one.sh ds_1.0 A '{"follower": "map"}' '{"t_clip_min": 0.65, "dec_scaler_for_steer": 1.0}' 256 25 oracle
bash $S/aur_one.sh asds_1.0 A '{"follower": "map"}' '{"t_clip_min": 0.65, "acc_scaler_for_steer": 1.0, "dec_scaler_for_steer": 1.0}' 256 25 oracle
touch $S/aur_q6.done
