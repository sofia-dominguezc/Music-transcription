import os, glob
import webbrowser
import json

# Assume get_notes is on your PYTHONPATH
get_notes = lambda: None

HTML_TEMPLATE = r'''
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Note Visualizer</title>
  <style>
    body { font-family: sans-serif; display: flex; flex-direction: column; align-items: center; }
    #controls { margin: 1em; }
    #canvas { border:1px solid #ccc; }
  </style>
</head>
<body>
  <div id="controls">
    <label for="songSelect">Song:</label>
    <select id="songSelect"></select>
    <button id="loadBtn">Load</button>
    <audio id="player" controls></audio>
  </div>
  <canvas id="canvas" width="600" height="600"></canvas>

  <script>
  // circle-of-fifths note order + hue anchors
  const noteOrder = ['C','G','D','A','E','B','F#','C#','Ab','Eb','Bb','F'];
  const anchorHues = {0:240, 4:0, 8:0};  // C=240°, E=0°, Ab=0°
  let hues = Array(12);
  const anchors = [...Object.keys(anchorHues).map(n=>+n).sort((a,b)=>a-b), 12];
  for (let i=0; i<anchors.length-1; i++) {
    const start = anchors[i], end = anchors[i+1];
    const h0 = anchorHues[start], h1 = (end===12? anchorHues[0] : anchorHues[end]);
    const dist = end>start? end-start : 12-start+end;
    for (let k=0; k<dist; k++) {
      const idx = (start + k) % 12;
      const frac = k / dist;
      const delta = ((h1 - h0 + 540) % 360) - 180;
      hues[idx] = (h0 + frac*delta + 360) % 360;
    }
  }

  const octaveCount = 8;
  const radiusMax = 280;
  const cx = 300, cy = 300;
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');

  function drawBase() {
    ctx.clearRect(0, 0, 600, 600);
    for (let o = 1; o <= octaveCount; o++) {
      const r1 = (o-1)/octaveCount * radiusMax;
      const r2 = o/octaveCount * radiusMax;
      // idle grayscale lightness: avoid full black/white -> [20%,80%]
      const light = 20 + ((o-1)/(octaveCount-1)) * 60;
      for (let i = 0; i < 12; i++) {
        const a1 = (i/12)*2*Math.PI - Math.PI/2;
        const a2 = ((i+1)/12)*2*Math.PI - Math.PI/2;
        ctx.beginPath();
        ctx.moveTo(cx + r1*Math.cos(a1), cy + r1*Math.sin(a1));
        ctx.arc(cx, cy, r2, a1, a2);
        ctx.arc(cx, cy, r1, a2, a1, true);
        ctx.fillStyle = `hsl(0,0%,${light}%)`;
        ctx.fill();
        ctx.stroke();
      }
    }
  }

  let notesData = [];
  function highlight(time) {
    drawBase();
    notesData.forEach(({start, end, midi}) => {
      if (time >= start && time <= end) {
        const n12 = midi % 12;
        const baseNames = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
        let name = baseNames[n12];
        if (name === 'C#') name = 'Db';
        if (name === 'D#') name = 'Eb';
        if (name === 'A#') name = 'Bb';
        const noteIdx = noteOrder.indexOf(name);
        const octave = Math.floor(midi/12) - 1;
        if (noteIdx < 0 || octave < 1 || octave > octaveCount) return;
        const r1 = (octave-1)/octaveCount * radiusMax;
        const r2 = octave/octaveCount * radiusMax;
        const a1 = (noteIdx/12)*2*Math.PI - Math.PI/2;
        const a2 = ((noteIdx+1)/12)*2*Math.PI - Math.PI/2;
        const hue = hues[noteIdx];
        // active lightness: avoid full white -> [60%,90%]
        const light = 60 + ((octave-1)/(octaveCount-1)) * 30;
        ctx.beginPath();
        ctx.moveTo(cx + r1*Math.cos(a1), cy + r1*Math.sin(a1));
        ctx.arc(cx, cy, r2, a1, a2);
        ctx.arc(cx, cy, r1, a2, a1, true);
        ctx.fillStyle = `hsl(${hue},100%,${light}%)`;
        ctx.fill();
      }
    });
  }

  window.onload = () => {
    drawBase();
    const songs = {{song_list}};
    const sel = document.getElementById('songSelect');
    songs.forEach(({name, ext}) => {
      const o = document.createElement('option');
      o.value = name; o.text = `${name}.${ext}`;
      sel.add(o);
    });
    document.getElementById('loadBtn').onclick = () => {
      const name = sel.value;
      const ext = songs.find(s => s.name === name).ext;
      const player = document.getElementById('player');
      player.src = `${name}.${ext}`;
      fetch(`${name}.csv`).then(r => r.text()).then(txt => {
        notesData = txt.split('\n').slice(1).filter(Boolean).map(line => {
          const [s, e, n] = line.split(',');
          return {start: +s, end: +e, midi: +n};
        });
      });
    };
    const player = document.getElementById('player');
    ['timeupdate', 'seeked'].forEach(evt =>
      player.addEventListener(evt, () => highlight(player.currentTime))
    );
    player.addEventListener('pause', drawBase);
  };
  </script>
</body>
</html>
'''


def main():
    # discover audio files
    audio_files = glob.glob('*.wav') + glob.glob('*.mp3')
    songs = []
    for f in audio_files:
        name, ext = os.path.splitext(f)
        songs.append({'name': name, 'ext': ext.lstrip('.')})

    # generate missing CSVs
    for s in songs:
        csv_file = f"{s['name']}.csv"
        if not os.path.exists(csv_file):
            print(f"Extracting notes for {s['name']}.{s['ext']}")
            get_notes(f"{s['name']}.{s['ext']}")

    # build HTML
    html = HTML_TEMPLATE.replace('{{song_list}}', json.dumps(songs))
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print('index.html written. Launching browser...')
    webbrowser.open('index.html')

if __name__ == '__main__':
    main()
