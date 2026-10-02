'use strict';
// End-to-end check of gesture control in the real app, no camera needed: feeds recorded hand
// poses into onHandResults and checks what the app plays.   npm run e2e
const path = require('path');
const { _electron: electron } = require('playwright-core');
const poses = require('../test/fixtures/e2e_hands.json');

async function main() {
  const env = { ...process.env };
  delete env.ELECTRON_RUN_AS_NODE;
  // An explicit executablePath stops Playwright injecting its loader, under which the renderer's
  // relative require('./theory') fails and the app never starts.
  const app = await electron.launch({ executablePath: require('electron'), args: [path.join(__dirname, '..')], env });
  const page = await app.firstWindow();
  const failures = [];
  const check = (name, ok, detail) => {
    console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : `  (${JSON.stringify(detail)})`}`);
    if (!ok) failures.push(name);
  };
  // spec: { note: {pose, at: note index} | null, quality: {pose} | null }; quality hand stays off the wheels.
  const feed = (spec, frames = 10) => page.evaluate(async ({ spec, frames, poses }) => {
    const notePoint = index => {
      const c = noteWheelCenter(), r = noteWheelRadius() * (1 + INNER_RATIO) / 2;
      const a = -Math.PI / 2 + index * (2 * Math.PI / 12);
      return { x: c.x + r * Math.cos(a), y: c.y + r * Math.sin(a) };
    };
    const place = (hand, pt) => {
      const idx = [0, 5, 9, 13, 17];
      const cx = idx.reduce((s, i) => s + hand[i][0], 0) / 5;
      const cy = idx.reduce((s, i) => s + hand[i][1], 0) / 5;
      const tx = 1 - pt.x / W, ty = pt.y / H;
      return hand.map(([x, y]) => ({ x: x - cx + tx, y: y - cy + ty, z: 0 }));
    };
    const hands = [], labels = [];
    if (spec.note) { hands.push(place(poses[spec.note.pose], notePoint(spec.note.at))); labels.push({ label: 'Right' }); }
    if (spec.quality) { hands.push(place(poses[spec.quality.pose], { x: W / 2, y: H * 0.12 })); labels.push({ label: 'Left' }); }
    for (let i = 0; i < frames; i++) {
      onHandResults({ multiHandLandmarks: hands, multiHandedness: labels });
      await new Promise(r => setTimeout(r, 40));
    }
    return { isPlaying, selectedNote, chord: document.getElementById('chord-name').textContent };
  }, { spec, frames, poses });

  try {
    await page.waitForFunction(() => window.__gesturesReady === true, null, { timeout: 20000 });
    await page.click('#gesture-btn');
    let s = await feed({ note: { pose: 'fist', at: 9 } });
    check('a fist over A plays it', s.isPlaying && s.selectedNote === 9, s);
    s = await feed({ note: { pose: 'palm', at: 9 } });
    check('an open palm releases it', !s.isPlaying, s);
    s = await feed({ note: { pose: 'fist', at: 9 }, quality: { pose: 'peace' } });
    check('a right-hand peace sign makes it A minor', s.isPlaying && s.chord.trim() === 'A min', s);
    await feed({ note: { pose: 'palm', at: 9 } });
    await page.evaluate(() => onNotePress(9, 0, 0));
    s = await feed({ note: { pose: 'fist', at: 9 } });
    check('a fist on a note already playing does not stop it', s.isPlaying && s.selectedNote === 9, s);
    s = await feed({}, 16);
    check('losing the hand releases the chord', !s.isPlaying, s);
  } finally {
    await app.close();
  }
  if (failures.length) { console.error(`${failures.length} check(s) failed`); process.exit(1); }
  console.log('all end-to-end checks passed');
}

main().catch(err => { console.error(err); process.exit(1); });
