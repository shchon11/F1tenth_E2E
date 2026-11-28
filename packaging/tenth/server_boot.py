"""설치본 서버의 시작점: `python -m tenth_boot f1sim.server --device cuda ...`

venv에 f1sim이 편집 설치(pip install -e)돼 있으면 그 찾기 고리(__editable__ finder)가 설치본 사본에 없는 모듈을
저장소 체크아웃에서 조용히 가져온다(2026-11-29 확인: 이 PC는 작업 트리, 연구실 PC는 옛 ~/F1tenth/F1tenth_E2E).
설치본은 자기 사본만 써야 하므로 그 고리를 떼고 대상 모듈을 __main__으로 돌린다. 없는 모듈은 분명한 ImportError가 된다.
"""
import runpy
import sys


def _editable(obj) -> bool:
    return "__editable__" in (getattr(obj, "__module__", "") or "") or "__editable__" in type(obj).__module__


sys.meta_path[:] = [f for f in sys.meta_path if not _editable(f)]
sys.path_hooks[:] = [h for h in sys.path_hooks if not _editable(h)]
sys.path[:] = [p for p in sys.path if "__editable__" not in p]
sys.path_importer_cache.clear()

if len(sys.argv) < 2:
    raise SystemExit("사용법: python -m tenth_boot <모듈> [인자...]")
target = sys.argv[1]
if target == "--where":                      # 실행기 check: 어느 사본·torch를 쓰는지
    import f1sim
    import torch
    print(f1sim.__file__, torch.__version__)
    raise SystemExit(0)
sys.argv = [target] + sys.argv[2:]
runpy.run_module(target, run_name="__main__", alter_sys=True)
