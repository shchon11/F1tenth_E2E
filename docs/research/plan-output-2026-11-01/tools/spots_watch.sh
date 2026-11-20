#!/bin/bash
# (11-20, 사용자: "저 이미지 추이들을 ... 사이트 거기 실시간에다가 계속 업로드를 해줘") 시험 충돌 지도를 사이트에 계속 올림:
# 1분마다 exam2 기록 중 아직 안 그린 것만 그림(tools/spots_frames.py) -> 실시간 페이지의 '시험 충돌 지도'
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
while true; do timeout 1800 python3 -W ignore spots_frames.py >> ../out/spots_watch.log 2>&1; sleep 60; done
