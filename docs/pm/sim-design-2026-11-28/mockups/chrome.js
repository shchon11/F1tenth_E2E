/* 목업 공용: 위 띠(내비게이션 + 현재 조합)와 아래 키 안내 띠를 그린다. */
(function () {
  const TABS = [['hub', '허브', '1'], ['garage', '차고', '2'], ['venue', '장소', '3'], ['studio', '트랙 스튜디오', '4'], ['race', '레이스', '5'], ['camp', '훈련소', '6'], ['lab', '분석', '7']];
  window.pwChrome = function (active, hints, diag) {
    const nav = document.createElement('header'); nav.className = 'nav';
    nav.innerHTML = `
      <div class="brand"><div class="mark">1/10</div><div class="word">TENTH<small>가칭</small></div></div>
      <nav class="nav-tabs">${TABS.map(t => `<a class="${t[0] === active ? 'on' : ''}"><span class="k">${t[2]}</span>${t[1]}</a>`).join('')}</nav>
      <div class="nav-right">
        <div class="loadout" title="현재 조합">
          <div class="slot"><span>차</span><b>실차 2호 · AGX Orin</b></div>
          <div class="slot"><span>장소</span><b>2026 한국 대회장</b></div>
          <div class="slot"><span>드라이버</span><b>미르 43</b></div>
        </div>
        <div class="nav-stat"><i class="dot lime"></i>훈련 2개 진행 중</div>
        <div class="nav-stat"><i class="dot"></i>연구실 PC<span class="dim2">·</span><i class="dot"></i>학습 서버</div>
      </div>`;
    document.body.appendChild(nav);
    const h = document.createElement('footer'); h.className = 'hints';
    h.innerHTML = hints.map(x => `<span><span class="key">${x[0]}</span>${x[1]}</span>`).join('') + `<span class="sp diag">${diag || ''}</span>`;
    document.body.appendChild(h);
  };
})();
