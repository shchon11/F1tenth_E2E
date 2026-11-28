// course outline as an inline SVG: the driving surface is the band between the two walls
function courseSVG(key, o) {
  const c = window.COURSES[key]; o = o || {};
  const pad = o.pad == null ? 0.2 : o.pad;
  let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${-pad} ${-pad} ${c.w + 2 * pad} ${c.h + 2 * pad}" preserveAspectRatio="xMidYMid meet" style="width:100%;height:100%;display:block">`;
  s += `<path d="${c.left} ${c.right}" fill="${o.fill || 'currentColor'}" fill-rule="evenodd"/>`;
  if (o.wall) s += `<path d="${c.left}" fill="none" stroke="${o.wall}" stroke-width="${o.wallW || 0.16}"/><path d="${c.right}" fill="none" stroke="${o.wall}" stroke-width="${o.wallW || 0.16}"/>`;
  if (o.start) s += `<g transform="translate(${c.start[0]} ${c.start[1]}) rotate(${c.start[2]})"><rect x="-0.09" y="-1.05" width="0.18" height="2.1" fill="${o.start}"/></g>`;
  if (o.boxes) c.boxes.forEach(b => s += `<rect x="${b[0] - 0.24}" y="${b[1] - 0.24}" width="0.48" height="0.48" fill="${o.boxes}"/>`);
  return s + '</svg>';
}
document.querySelectorAll('[data-course]').forEach(el => {
  el.innerHTML = courseSVG(el.dataset.course, {fill: el.dataset.fill, wall: el.dataset.wall, start: el.dataset.start, boxes: el.dataset.boxes, wallW: el.dataset.wallw});
});
