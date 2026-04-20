// Cast — simplified, friendly UI. Single file for clarity.

const { useState, useEffect, useMemo, useRef, useCallback } = React;
const { VIDEO, ENTITIES, CLIPS, EXAMPLE_QUERIES } = window.CAST_DATA;

// ───────── helpers ─────────
function fmtTime(sec) {
  sec = Math.max(0, Math.floor(sec));
  const m = Math.floor(sec / 60), s = sec % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}
function FaceCrop({ entity, size = 32, ring = false }) {
  const h = entity.hue;
  return (
    <div style={{
      width: size, height: size, borderRadius: "50%", flexShrink: 0,
      background: `radial-gradient(ellipse 58% 72% at 50% 55%, oklch(0.8 0.08 ${h}) 0%, oklch(0.55 0.1 ${h}) 60%, oklch(0.35 0.06 ${h}) 100%)`,
      boxShadow: ring ? `0 0 0 2px white, 0 0 0 4px var(--accent)` : `0 0 0 1px oklch(1 0 0 / 0.5)`,
      position: "relative", overflow: "hidden",
    }}>
      <div style={{ position: "absolute", top: size*0.38, left: size*0.3, width: size*0.1, height: size*0.07, background: "oklch(0.15 0 0)", borderRadius: 1 }} />
      <div style={{ position: "absolute", top: size*0.38, right: size*0.3, width: size*0.1, height: size*0.07, background: "oklch(0.15 0 0)", borderRadius: 1 }} />
    </div>
  );
}

// ───────── fake retrieval ─────────
function runQuery(q) {
  const text = (q.text || "").toLowerCase().trim();
  const tokens = text.split(/\s+/).filter(t => t.length > 2);
  const EXPAND = {
    lied: ["lied", "lie", "deceive", "betray", "copied", "truth"],
    realize: ["realize", "realise", "discover", "learn", "understand"],
    chart: ["chart", "map"],
    taken: ["taken", "seize", "slid", "unroll", "hand", "hauls"],
    storm: ["storm", "fog"],
    loyalty: ["loyalty", "allegiance"],
  };
  const expanded = new Set(tokens);
  tokens.forEach(t => Object.entries(EXPAND).forEach(([,syns]) => {
    if (syns.some(s => t.includes(s) || s.includes(t))) syns.forEach(s => expanded.add(s));
  }));

  const bm25 = CLIPS.map(c => {
    const blob = (c.transcript + " " + c.caption).toLowerCase();
    let s = 0;
    expanded.forEach(t => { if (blob.includes(t)) s += tokens.includes(t) ? 1.2 : 0.6; });
    return { clip: c, raw: s };
  }).sort((a,b)=>b.raw-a.raw);

  const vec = CLIPS.map(c => {
    const cap = c.caption.toLowerCase();
    let s = 0;
    expanded.forEach(t => { if (cap.includes(t)) s += 0.9; });
    if (text.includes("lied") || text.includes("realize")) {
      if (cap.match(/copied|journal|seal|glyphs|truth/)) s += 1.4;
    }
    return { clip: c, raw: s };
  }).sort((a,b)=>b.raw-a.raw);

  const face = CLIPS.map(c => {
    let s = 0;
    if (q.entity && c.present.includes(q.entity)) s += 1;
    if (q.entity2 && c.present.includes(q.entity2)) s += 1;
    if (q.entity && q.entity2 && c.present.includes(q.entity) && c.present.includes(q.entity2)) s += 0.5;
    return { clip: c, raw: s };
  }).sort((a,b)=>b.raw-a.raw);

  const graph = CLIPS.map(c => {
    let s = 0;
    if (q.predicate && q.entity) {
      const st = c.states[q.entity];
      if (st) {
        if (q.predicate === "loyalty:shift") {
          const first = CLIPS.find(cc => cc.states[q.entity])?.states[q.entity];
          if (first && st.loyalty && st.loyalty !== first.loyalty) s += 1.2;
        } else if (q.predicate.startsWith("feeling:")) {
          const w = q.predicate.split(":")[1];
          if (st.feeling && st.feeling.includes(w)) s += 1;
        }
      }
    }
    return { clip: c, raw: s };
  }).sort((a,b)=>b.raw-a.raw);

  const K = 60;
  const weights = {
    words:   tokens.length > 0 ? 1 : 0,
    meaning: tokens.length > 0 ? 1 : 0,
    people:  (q.entity || q.entity2) ? 1.3 : 0,
    state:   q.predicate ? 1.3 : 0,
  };
  const ranks = { words:{}, meaning:{}, people:{}, state:{} };
  [["words", bm25], ["meaning", vec], ["people", face], ["state", graph]].forEach(([name, list]) => {
    list.forEach((it, i) => { if (it.raw > 0) ranks[name][it.clip.idx] = i; });
  });
  return CLIPS.map(c => {
    const breakdown = [];
    let score = 0;
    ["words","meaning","people","state"].forEach(src => {
      if (c.idx in ranks[src] && weights[src] > 0) {
        const r = ranks[src][c.idx];
        score += weights[src] / (K + r);
        breakdown.push({ src, rank: r + 1 });
      }
    });
    return { clip: c, score, breakdown };
  }).filter(r => r.score > 0).sort((a,b) => b.score - a.score).slice(0, 6);
}

// ───────── UI ─────────
function Header({ onUpload }) {
  return (
    <header style={{
      borderBottom: "1px solid var(--line)", background: "white",
      padding: "18px 32px", display: "flex", alignItems: "center", gap: 16,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{
          width: 28, height: 28, borderRadius: "50%",
          background: "var(--accent)",
          position: "relative",
        }}>
          <div style={{ position: "absolute", top: 7, left: 7, width: 14, height: 14, borderRadius: "50%", border: "2px solid white" }} />
        </div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 16, letterSpacing: -0.3 }}>Cast</div>
          <div style={{ fontSize: 11, color: "var(--fg-mut)" }}>Search videos by who's on screen and what's happening</div>
        </div>
      </div>
      <div style={{ flex: 1 }} />
      <nav style={{ display: "flex", gap: 22, fontSize: 13, color: "var(--fg-mut)" }}>
        <span style={{ color: "var(--fg)", fontWeight: 500, borderBottom: "2px solid var(--accent)", paddingBottom: 4 }}>Library</span>
        <span onClick={onUpload} style={{ cursor: "pointer" }}>Upload</span>
        <span>Settings</span>
      </nav>
    </header>
  );
}

function VideoCard({ video, cast, activeIdx, clip }) {
  return (
    <section style={{
      background: "white", border: "1px solid var(--line)", borderRadius: 10, overflow: "hidden",
    }}>
      <div style={{ padding: "20px 24px 16px" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 2 }}>
          <h1 style={{ fontFamily: "var(--serif)", fontSize: 30, fontStyle: "italic", fontWeight: 400, letterSpacing: -0.5 }}>
            {video.title}
          </h1>
          <span style={{ color: "var(--fg-mut)", fontSize: 14 }}>{video.subtitle}</span>
        </div>
        <div style={{ fontSize: 12, color: "var(--fg-dim)" }}>
          {fmtTime(video.duration_sec)} · {CLIPS.length} scenes · {cast.length} people identified · Added {video.ingested_at}
        </div>
      </div>

      {/* faux video */}
      <div style={{ position: "relative", aspectRatio: "16/9", background: "oklch(0.2 0.01 60)", overflow: "hidden" }}>
        <div style={{
          position: "absolute", inset: 0,
          background: `repeating-linear-gradient(135deg, oklch(0.22 0.01 60) 0 18px, oklch(0.18 0.01 60) 18px 36px)`,
        }} />
        <div style={{ position: "absolute", top: 16, left: 20, color: "oklch(0.7 0 0)", fontSize: 11, letterSpacing: 0.5 }}>
          Scene {clip.idx + 1} of {CLIPS.length} · {fmtTime(clip.start)} – {fmtTime(clip.end)}
        </div>
        <div style={{
          position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)",
          width: 64, height: 64, borderRadius: "50%",
          background: "oklch(1 0 0 / 0.15)", backdropFilter: "blur(6px)",
          border: "1px solid oklch(1 0 0 / 0.5)",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <div style={{ width: 0, height: 0, borderTop: "10px solid transparent", borderBottom: "10px solid transparent", borderLeft: "16px solid white", marginLeft: 4 }} />
        </div>
        <div style={{
          position: "absolute", bottom: 0, left: 0, right: 0, padding: "18px 20px 16px",
          background: "linear-gradient(180deg, transparent, oklch(0 0 0 / 0.7))",
          color: "white",
        }}>
          <div style={{ fontFamily: "var(--serif)", fontStyle: "italic", fontSize: 16, marginBottom: 4, textWrap: "balance" }}>
            "{clip.transcript.replace(/^—\s*/, "").split("—")[0].trim()}"
          </div>
          <div style={{ fontSize: 12, color: "oklch(0.85 0 0)", textWrap: "pretty" }}>
            {clip.caption}
          </div>
        </div>
      </div>

      {/* present cast */}
      <div style={{ padding: "14px 24px", display: "flex", alignItems: "center", gap: 12, borderTop: "1px solid var(--line)" }}>
        <span style={{ fontSize: 12, color: "var(--fg-mut)" }}>In this scene:</span>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {cast.filter(e => clip.present.includes(e.id)).map(e => (
            <div key={e.id} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <FaceCrop entity={e} size={22} />
              <span style={{ fontSize: 13 }}>{e.label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Timeline({ clips, activeIdx, onSelect, highlighted }) {
  const total = clips[clips.length - 1].end;
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <h3 style={{ fontSize: 14 }}>All scenes</h3>
        <span style={{ fontSize: 12, color: "var(--fg-mut)" }}>Click any scene to jump to it</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 8 }}>
        {clips.map(c => {
          const active = c.idx === activeIdx;
          const hit = highlighted && highlighted.has(c.idx);
          return (
            <button key={c.idx} onClick={() => onSelect(c.idx)}
              style={{
                textAlign: "left", padding: 10, borderRadius: 8,
                background: active ? "var(--accent-soft)" : (hit ? "oklch(0.97 0.02 45)" : "white"),
                border: active ? "1.5px solid var(--accent)" : `1px solid ${hit ? "oklch(0.85 0.04 45)" : "var(--line)"}`,
                cursor: "pointer", transition: "background 120ms",
              }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: active ? "var(--accent)" : "var(--fg-mut)" }}>
                  Scene {c.idx + 1}
                </span>
                <span style={{ fontSize: 10, color: "var(--fg-dim)" }}>{fmtTime(c.start)}</span>
              </div>
              <div style={{ fontSize: 11, color: "var(--fg)", lineHeight: 1.35, minHeight: 28, textWrap: "pretty" }}>
                {c.caption.split(".")[0]}.
              </div>
              <div style={{ display: "flex", gap: 3, marginTop: 8 }}>
                {c.present.map(id => {
                  const e = ENTITIES.find(x => x.id === id);
                  return e ? <FaceCrop key={id} entity={e} size={16} /> : null;
                })}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function CastList({ cast, clips, selected, onSelect }) {
  return (
    <div>
      <h3 style={{ fontSize: 14, marginBottom: 10 }}>Cast <span style={{ color: "var(--fg-mut)", fontWeight: 400, fontSize: 12 }}>· recognized from faces</span></h3>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {cast.map(e => {
          const n = clips.filter(c => c.present.includes(e.id)).length;
          const sel = selected === e.id;
          return (
            <button key={e.id} onClick={() => onSelect(sel ? null : e.id)}
              style={{
                display: "flex", alignItems: "center", gap: 12, textAlign: "left",
                padding: "8px 10px", borderRadius: 8,
                border: sel ? "1.5px solid var(--accent)" : "1px solid transparent",
                background: sel ? "var(--accent-soft)" : "transparent",
              }}>
              <FaceCrop entity={e} size={36} ring={sel} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 14, fontWeight: 500 }}>{e.label}</div>
                <div style={{ fontSize: 11, color: "var(--fg-mut)" }}>
                  {e.role ?? "Unnamed"} · {n} scene{n !== 1 ? "s" : ""}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function StateTimeline({ entity, clips }) {
  if (!entity) return (
    <div style={{ padding: "28px 16px", textAlign: "center", color: "var(--fg-mut)", fontSize: 13, background: "var(--bg-alt)", borderRadius: 8 }}>
      Select someone from the Cast to see how they change over time.
    </div>
  );

  const dims = [
    { key: "feeling",  label: "How they feel",    color: "oklch(0.55 0.15 35)" },
    { key: "knows",    label: "What they know",   color: "oklch(0.52 0.1 210)" },
    { key: "loyalty",  label: "Who they trust",   color: "oklch(0.5 0.13 330)" },
    { key: "goal",     label: "What they want",   color: "oklch(0.5 0.1 155)" },
  ];

  const changes = dims.map(d => {
    const events = [];
    let last = null;
    clips.forEach(c => {
      const v = c.states[entity.id]?.[d.key];
      if (v && v !== last) {
        events.push({ clip: c, value: v, changed: last !== null });
        last = v;
      }
    });
    return { ...d, events };
  });

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
        <FaceCrop entity={entity} size={28} />
        <div>
          <div style={{ fontSize: 14, fontWeight: 600 }}>How {entity.label} changes through the episode</div>
          <div style={{ fontSize: 11, color: "var(--fg-mut)" }}>Extracted by the language model from transcript and scene context</div>
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {changes.map(d => (
          <div key={d.key}>
            <div style={{ fontSize: 11, color: "var(--fg-mut)", marginBottom: 6, textTransform: "uppercase", letterSpacing: 0.4 }}>{d.label}</div>
            {d.events.length === 0 ? (
              <div style={{ fontSize: 12, color: "var(--fg-dim)", fontStyle: "italic" }}>no changes recorded</div>
            ) : (
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
                {d.events.map((ev, i) => (
                  <React.Fragment key={i}>
                    <span title={`Scene ${ev.clip.idx + 1} · ${fmtTime(ev.clip.start)}`}
                      style={{
                        fontSize: 13,
                        padding: "4px 10px",
                        borderRadius: 999,
                        background: "white",
                        border: `1px solid ${d.color}`,
                        color: d.color,
                        fontFamily: "var(--serif)", fontStyle: "italic",
                      }}>
                      {ev.value}
                    </span>
                    {i < d.events.length - 1 && (
                      <span style={{ color: "var(--fg-dim)", fontSize: 11 }}>→</span>
                    )}
                  </React.Fragment>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function SearchPanel({ query, setQuery, onRun, running, cast, examples }) {
  const predicates = [
    { id: null,            label: "Any" },
    { id: "loyalty:shift", label: "Loyalty changes" },
    { id: "feeling:afraid",label: "Feels afraid" },
    { id: "feeling:resolute", label: "Feels resolute" },
  ];

  return (
    <div style={{ background: "white", border: "1px solid var(--line)", borderRadius: 10, padding: 18 }}>
      <h3 style={{ fontSize: 14, marginBottom: 12 }}>Search</h3>

      <label style={{ fontSize: 11, color: "var(--fg-mut)", display: "block", marginBottom: 4 }}>What happens in the scene</label>
      <textarea
        value={query.text}
        onChange={e => setQuery({ ...query, text: e.target.value })}
        onKeyDown={e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) onRun(); }}
        placeholder="e.g. someone realizes they've been lied to"
        rows={2}
        style={{
          width: "100%", padding: "8px 10px", fontSize: 13,
          border: "1px solid var(--line)", borderRadius: 6,
          resize: "none", outline: "none",
          fontFamily: "var(--serif)", fontStyle: "italic", lineHeight: 1.4,
        }}
      />

      <label style={{ fontSize: 11, color: "var(--fg-mut)", display: "block", margin: "12px 0 6px" }}>Who's in it</label>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        <button onClick={() => setQuery({ ...query, entity: null, entity2: null })}
          style={chipStyle(query.entity === null && !query.entity2)}>Anyone</button>
        {cast.map(e => (
          <button key={e.id}
            onClick={() => {
              if (query.entity === e.id) setQuery({ ...query, entity: null });
              else if (query.entity2 === e.id) setQuery({ ...query, entity2: null });
              else if (!query.entity) setQuery({ ...query, entity: e.id });
              else setQuery({ ...query, entity2: e.id });
            }}
            style={{
              ...chipStyle(query.entity === e.id || query.entity2 === e.id),
              display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 10px 4px 5px",
            }}>
            <FaceCrop entity={e} size={18} />
            {e.label.split(" ")[0]}
          </button>
        ))}
      </div>

      <label style={{ fontSize: 11, color: "var(--fg-mut)", display: "block", margin: "12px 0 6px" }}>
        Their state {!query.entity && <span style={{ color: "var(--fg-dim)" }}>(pick someone first)</span>}
      </label>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", opacity: query.entity ? 1 : 0.5 }}>
        {predicates.map(p => (
          <button key={String(p.id)} onClick={() => setQuery({ ...query, predicate: p.id })}
            disabled={!query.entity}
            style={chipStyle(query.predicate === p.id)}>
            {p.label}
          </button>
        ))}
      </div>

      <button onClick={onRun} disabled={running}
        style={{
          marginTop: 16, width: "100%", padding: "10px 14px",
          background: "var(--accent)", color: "white", border: "none", borderRadius: 6,
          fontSize: 14, fontWeight: 600,
        }}>
        {running ? "Searching…" : "Search"}
      </button>

      <div style={{ marginTop: 14, paddingTop: 14, borderTop: "1px solid var(--line)" }}>
        <div style={{ fontSize: 11, color: "var(--fg-mut)", marginBottom: 6 }}>Try an example</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {examples.map((ex, i) => (
            <button key={i}
              onClick={() => setQuery({ text: ex.text, entity: ex.entity, entity2: ex.entity2 || null, predicate: ex.predicate })}
              style={{
                textAlign: "left", padding: "6px 8px", fontSize: 12,
                border: "none", background: "transparent", color: "var(--fg)",
                borderRadius: 4, cursor: "pointer",
              }}
              onMouseEnter={e => e.currentTarget.style.background = "var(--bg-alt)"}
              onMouseLeave={e => e.currentTarget.style.background = "transparent"}
            >
              → {ex.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
function chipStyle(active) {
  return {
    padding: "4px 10px", fontSize: 12, borderRadius: 999,
    border: `1px solid ${active ? "var(--accent)" : "var(--line)"}`,
    background: active ? "var(--accent-soft)" : "white",
    color: active ? "var(--accent)" : "var(--fg)",
    cursor: "pointer", fontWeight: active ? 600 : 400,
  };
}

function Results({ results, onSelect, activeIdx, query }) {
  if (results === null) return null;
  if (results.length === 0) return (
    <div style={{ padding: 20, textAlign: "center", fontSize: 13, color: "var(--fg-mut)" }}>
      No scenes matched. Try removing a filter.
    </div>
  );
  const tokens = (query.text || "").toLowerCase().split(/\s+/).filter(t => t.length > 3);
  const reasonLabel = { words: "word match", meaning: "meaning", people: "who's on screen", state: "their state" };

  return (
    <div>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 10 }}>
        <h3 style={{ fontSize: 14 }}>Best matches</h3>
        <span style={{ fontSize: 12, color: "var(--fg-mut)" }}>{results.length} scenes</span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {results.map((r, i) => {
          const c = r.clip;
          const active = c.idx === activeIdx;
          return (
            <button key={c.idx} onClick={() => onSelect(c.idx)}
              style={{
                textAlign: "left", padding: 14, borderRadius: 8,
                background: active ? "var(--accent-soft)" : "white",
                border: `1px solid ${active ? "var(--accent)" : "var(--line)"}`,
                display: "flex", gap: 12,
              }}>
              <div style={{
                width: 36, height: 36, flexShrink: 0,
                borderRadius: 8, background: "var(--accent)", color: "white",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 14, fontWeight: 700,
              }}>{i + 1}</div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 3 }}>
                  <span style={{ fontSize: 13, fontWeight: 600 }}>Scene {c.idx + 1}</span>
                  <span style={{ fontSize: 11, color: "var(--fg-mut)" }}>{fmtTime(c.start)} – {fmtTime(c.end)}</span>
                </div>
                <div style={{ fontSize: 13, marginBottom: 6, lineHeight: 1.4, textWrap: "pretty" }}>
                  <Highlight text={c.caption} tokens={tokens} />
                </div>
                <div style={{ fontSize: 12, color: "var(--fg-mut)", fontFamily: "var(--serif)", fontStyle: "italic", lineHeight: 1.4, marginBottom: 8 }}>
                  <Highlight text={c.transcript} tokens={tokens} />
                </div>
                <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
                  <span style={{ fontSize: 10, color: "var(--fg-dim)" }}>Matched on:</span>
                  {r.breakdown.map(b => (
                    <span key={b.src} style={{
                      fontSize: 10, padding: "1px 7px", borderRadius: 4,
                      background: "var(--bg-alt)", color: "var(--fg-mut)",
                      border: "1px solid var(--line)",
                    }}>{reasonLabel[b.src]}</span>
                  ))}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
function Highlight({ text, tokens }) {
  if (!tokens.length) return text;
  const re = new RegExp(`(${tokens.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join("|")})`, "gi");
  const parts = text.split(re);
  return parts.map((p, i) => tokens.some(t => p.toLowerCase() === t.toLowerCase()) ? <mark key={i}>{p}</mark> : <span key={i}>{p}</span>);
}

function HelpBanner({ onDismiss, visible }) {
  if (!visible) return null;
  return (
    <div style={{
      background: "var(--accent-soft)", border: "1px solid oklch(0.85 0.06 45)",
      borderRadius: 10, padding: "14px 18px",
      display: "flex", gap: 14, alignItems: "flex-start",
    }}>
      <div style={{
        width: 28, height: 28, borderRadius: "50%", background: "var(--accent)",
        color: "white", display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 14, fontWeight: 700, flexShrink: 0,
      }}>?</div>
      <div style={{ flex: 1 }}>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>What is Cast?</div>
        <div style={{ fontSize: 13, color: "var(--fg)", lineHeight: 1.5, textWrap: "pretty" }}>
          Cast watches your videos and figures out <b>who</b> appears in each scene (by recognizing faces)
          and <b>what</b> is happening (by reading the transcript and looking at each frame).
          You can then search across everything — not just by words, but by <i>who's on screen</i> and
          how their feelings, loyalties, or goals change over time.
        </div>
      </div>
      <button onClick={onDismiss}
        style={{ border: "none", background: "transparent", color: "var(--fg-mut)", fontSize: 18, lineHeight: 1 }}>×</button>
    </div>
  );
}

// ───────── upload landing ─────────
function UploadLanding({ onStart, accent }) {
  const [dragOver, setDragOver] = useState(false);
  const [file, setFile] = useState(null);
  const fileInput = useRef(null);

  const onPick = (f) => {
    if (!f) return;
    setFile({ name: f.name, size: f.size });
  };

  const SAMPLE_URLS = [
    "The Cartographer's Apprentice — Episode 03",
    "Onboarding training · Acme Co · Week 2",
    "All-hands meeting · Oct 14 recording",
  ];

  return (
    <div style={{ maxWidth: 880, margin: "0 auto", padding: "64px 32px 48px" }}>
      <div style={{ textAlign: "center", marginBottom: 32 }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
          <div style={{
            width: 40, height: 40, borderRadius: "50%", background: "var(--accent)", position: "relative",
          }}>
            <div style={{ position: "absolute", top: 10, left: 10, width: 20, height: 20, borderRadius: "50%", border: "2.5px solid white" }} />
          </div>
          <span style={{ fontWeight: 700, fontSize: 22, letterSpacing: -0.4 }}>Cast</span>
        </div>
        <h1 style={{ fontFamily: "var(--serif)", fontStyle: "italic", fontWeight: 400, fontSize: 44, letterSpacing: -0.8, marginBottom: 12, textWrap: "balance" }}>
          Search videos by who's in them and what happens.
        </h1>
        <p style={{ fontSize: 16, color: "var(--fg-mut)", lineHeight: 1.5, maxWidth: 620, margin: "0 auto", textWrap: "pretty" }}>
          Upload a video. Cast transcribes it, recognizes every face across scenes, and
          tracks how each person's feelings, loyalties, and goals change over time —
          so you can find moments, not just keywords.
        </p>
      </div>

      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault(); setDragOver(false);
          const f = e.dataTransfer.files?.[0];
          if (f) onPick(f);
        }}
        onClick={() => fileInput.current?.click()}
        style={{
          border: `2px dashed ${dragOver ? "var(--accent)" : "oklch(0.82 0.01 60)"}`,
          background: dragOver ? "var(--accent-soft)" : "white",
          borderRadius: 14,
          padding: "40px 32px",
          textAlign: "center",
          cursor: "pointer",
          transition: "border-color 120ms, background 120ms",
        }}>
        <input
          ref={fileInput} type="file" accept="video/*" hidden
          onChange={(e) => onPick(e.target.files?.[0])}
        />
        <div style={{
          width: 52, height: 52, margin: "0 auto 14px",
          borderRadius: "50%", background: "var(--accent-soft)",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 3v14M5 10l7-7 7 7M5 21h14" />
          </svg>
        </div>
        {!file ? (
          <>
            <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 4 }}>
              Drop a video here, or click to choose a file
            </div>
            <div style={{ fontSize: 13, color: "var(--fg-mut)" }}>
              Supports .mp4 · .mov · .mkv · .webm up to 8 GB
            </div>
          </>
        ) : (
          <>
            <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 4 }}>{file.name}</div>
            <div style={{ fontSize: 12, color: "var(--fg-mut)" }}>
              {(file.size / (1024 * 1024)).toFixed(1)} MB — ready to process
            </div>
          </>
        )}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 12, margin: "24px 0" }}>
        <div style={{ flex: 1, height: 1, background: "var(--line)" }} />
        <span style={{ fontSize: 12, color: "var(--fg-dim)" }}>or paste a link</span>
        <div style={{ flex: 1, height: 1, background: "var(--line)" }} />
      </div>

      <div style={{ display: "flex", gap: 8 }}>
        <input
          type="url" placeholder="https://… (YouTube, Vimeo, or direct video URL)"
          style={{
            flex: 1, padding: "12px 14px", fontSize: 14,
            border: "1px solid var(--line)", borderRadius: 8, outline: "none",
            background: "white",
          }}
          onFocus={(e) => e.target.style.borderColor = "var(--accent)"}
          onBlur={(e) => e.target.style.borderColor = "var(--line)"}
        />
        <button
          onClick={() => onStart({ title: file?.name?.replace(/\.[^.]+$/, "") || "Your video" })}
          style={{
            padding: "12px 22px",
            background: "var(--accent)", color: "white", border: "none", borderRadius: 8,
            fontSize: 14, fontWeight: 600,
          }}>
          Analyze
        </button>
      </div>

      <div style={{ marginTop: 40 }}>
        <div style={{ fontSize: 12, color: "var(--fg-mut)", marginBottom: 10, textAlign: "center" }}>
          Or try a sample video
        </div>
        <div style={{ display: "flex", gap: 8, justifyContent: "center", flexWrap: "wrap" }}>
          {SAMPLE_URLS.map((s, i) => (
            <button key={i}
              onClick={() => onStart({ title: s, sample: i === 0 })}
              style={{
                padding: "8px 14px", fontSize: 13,
                background: "white", border: "1px solid var(--line)", borderRadius: 999,
                color: "var(--fg)",
              }}>
              {s}
            </button>
          ))}
        </div>
      </div>

      <div style={{ marginTop: 56, display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 20 }}>
        {[
          { n: "1", t: "Upload", d: "Your video is transcribed and split into scenes automatically." },
          { n: "2", t: "Understand", d: "Cast recognizes faces across every scene and reads what's happening." },
          { n: "3", t: "Search", d: "Ask questions about moments, people, and how they change." },
        ].map(s => (
          <div key={s.n} style={{ textAlign: "center", padding: "0 8px" }}>
            <div style={{
              width: 32, height: 32, margin: "0 auto 8px", borderRadius: "50%",
              background: "var(--accent-soft)", color: "var(--accent)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 14, fontWeight: 700,
            }}>{s.n}</div>
            <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 3 }}>{s.t}</div>
            <div style={{ fontSize: 12, color: "var(--fg-mut)", lineHeight: 1.45 }}>{s.d}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ───────── processing screen ─────────
function ProcessingScreen({ title, onDone }) {
  const STEPS = [
    { label: "Reading the audio track",    detail: "Transcribing speech with Whisper" },
    { label: "Splitting into scenes",       detail: "Detecting cuts and extracting keyframes" },
    { label: "Finding faces",               detail: "Running face detection on each scene" },
    { label: "Grouping people",             detail: "Clustering faces into distinct characters" },
    { label: "Understanding what happens",  detail: "Describing each scene and extracting states" },
    { label: "Almost done",                 detail: "Building your searchable library" },
  ];
  const [step, setStep] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (step >= STEPS.length) { onDone(); return; }
    const t = setInterval(() => {
      setProgress(p => {
        const next = p + (2 + Math.random() * 4);
        if (next >= 100) {
          clearInterval(t);
          setTimeout(() => { setStep(s => s + 1); setProgress(0); }, 120);
          return 100;
        }
        return next;
      });
    }, 60);
    return () => clearInterval(t);
  }, [step]);

  return (
    <div style={{ maxWidth: 640, margin: "0 auto", padding: "80px 32px" }}>
      <div style={{ textAlign: "center", marginBottom: 36 }}>
        <div style={{
          width: 56, height: 56, margin: "0 auto 18px",
          borderRadius: "50%", border: "3px solid var(--accent-soft)",
          borderTopColor: "var(--accent)",
          animation: "spin 1s linear infinite",
        }} />
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        <h1 style={{ fontFamily: "var(--serif)", fontStyle: "italic", fontWeight: 400, fontSize: 30, marginBottom: 6 }}>
          Analyzing your video
        </h1>
        <div style={{ color: "var(--fg-mut)", fontSize: 14 }}>{title}</div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {STEPS.map((s, i) => {
          const done = i < step;
          const active = i === step;
          return (
            <div key={i} style={{
              display: "flex", gap: 14, alignItems: "center",
              padding: "12px 16px", borderRadius: 8,
              background: active ? "white" : "transparent",
              border: active ? "1px solid var(--line)" : "1px solid transparent",
              opacity: done ? 0.6 : 1,
            }}>
              <div style={{
                width: 22, height: 22, borderRadius: "50%", flexShrink: 0,
                background: done ? "var(--accent)" : (active ? "var(--accent-soft)" : "oklch(0.95 0.005 60)"),
                color: done ? "white" : "var(--accent)",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 11, fontWeight: 700,
              }}>
                {done ? "✓" : (active ? <span style={{ animation: "pulse 1.2s infinite" }}>●</span> : i + 1)}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 14, fontWeight: active ? 600 : 500 }}>{s.label}</div>
                <div style={{ fontSize: 11, color: "var(--fg-mut)" }}>{s.detail}</div>
                {active && (
                  <div style={{ height: 3, background: "var(--accent-soft)", borderRadius: 3, marginTop: 6, overflow: "hidden" }}>
                    <div style={{ width: `${progress}%`, height: "100%", background: "var(--accent)", transition: "width 60ms linear" }} />
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ───────── main app ─────────
function CastApp() {
  const TWEAKS = /*EDITMODE-BEGIN*/{ "accent": "0.56 0.16 35", "showHelp": true }/*EDITMODE-END*/;
  const [tweaks, setTweaks] = useState(TWEAKS);
  const [tweaksOpen, setTweaksOpen] = useState(false);

  useEffect(() => {
    const onMsg = e => {
      if (e.data?.type === "__activate_edit_mode") setTweaksOpen(true);
      else if (e.data?.type === "__deactivate_edit_mode") setTweaksOpen(false);
    };
    window.addEventListener("message", onMsg);
    window.parent.postMessage({ type: "__edit_mode_available" }, "*");
    return () => window.removeEventListener("message", onMsg);
  }, []);
  const persist = edits => {
    setTweaks(t => ({ ...t, ...edits }));
    window.parent.postMessage({ type: "__edit_mode_set_keys", edits }, "*");
  };

  const [view, setView] = useState(() => localStorage.getItem("cast.view") || "upload");
  const [pendingTitle, setPendingTitle] = useState(VIDEO.title);
  useEffect(() => { localStorage.setItem("cast.view", view); }, [view]);

  const [activeIdx, setActiveIdx] = useState(() => parseInt(localStorage.getItem("cast.idx") || "5", 10));
  useEffect(() => { localStorage.setItem("cast.idx", String(activeIdx)); }, [activeIdx]);
  useEffect(() => { localStorage.setItem("cast.idx", String(activeIdx)); }, [activeIdx]);

  const [selected, setSelected] = useState("e1");
  const [query, setQueryState] = useState({ text: "someone realizes they've been lied to", entity: "e1", entity2: null, predicate: "loyalty:shift" });
  const [results, setResults] = useState(null);
  const [running, setRunning] = useState(false);
  const [helpVisible, setHelpVisible] = useState(tweaks.showHelp);

  useEffect(() => setHelpVisible(tweaks.showHelp), [tweaks.showHelp]);

  const doRun = useCallback(() => {
    setRunning(true);
    setTimeout(() => {
      const r = runQuery(query);
      setResults(r);
      setRunning(false);
      if (r.length > 0) setActiveIdx(r[0].clip.idx);
    }, 280);
  }, [query]);

  useEffect(() => { doRun(); /* eslint-disable-line */ }, []);

  const highlighted = useMemo(() => results ? new Set(results.map(r => r.clip.idx)) : null, [results]);
  const activeClip = CLIPS[activeIdx];
  const selEntity = ENTITIES.find(e => e.id === selected);

  const rootStyle = { "--accent": `oklch(${tweaks.accent})` };

  if (view === "upload") {
    return (
      <div style={{ ...rootStyle, minHeight: "100vh" }}>
        <UploadLanding onStart={(info) => {
          setPendingTitle(info.title);
          setView("processing");
        }} />
      </div>
    );
  }
  if (view === "processing") {
    return (
      <div style={{ ...rootStyle, minHeight: "100vh" }}>
        <ProcessingScreen title={pendingTitle} onDone={() => setView("analysis")} />
      </div>
    );
  }

  return (
    <div style={{ ...rootStyle, minHeight: "100vh" }}>
      <Header onUpload={() => setView("upload")} />
      <div style={{ maxWidth: 1240, margin: "0 auto", padding: "24px 32px 48px", display: "grid", gridTemplateColumns: "1fr 340px", gap: 24 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <HelpBanner visible={helpVisible} onDismiss={() => setHelpVisible(false)} />
          <VideoCard video={VIDEO} cast={ENTITIES} activeIdx={activeIdx} clip={activeClip} />
          <section style={{ background: "white", border: "1px solid var(--line)", borderRadius: 10, padding: 20 }}>
            <Timeline clips={CLIPS} activeIdx={activeIdx} onSelect={setActiveIdx} highlighted={highlighted} />
          </section>
          <section style={{ background: "white", border: "1px solid var(--line)", borderRadius: 10, padding: 20 }}>
            <StateTimeline entity={selEntity} clips={CLIPS} />
          </section>
        </div>

        <aside style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div style={{ background: "white", border: "1px solid var(--line)", borderRadius: 10, padding: 18 }}>
            <CastList cast={ENTITIES} clips={CLIPS} selected={selected} onSelect={setSelected} />
          </div>
          <SearchPanel query={query} setQuery={setQueryState} onRun={doRun} running={running} cast={ENTITIES} examples={EXAMPLE_QUERIES} />
          <div style={{ background: "white", border: "1px solid var(--line)", borderRadius: 10, padding: 18 }}>
            <Results results={results} onSelect={setActiveIdx} activeIdx={activeIdx} query={query} />
          </div>
        </aside>
      </div>

      {tweaksOpen && (
        <div style={{
          position: "fixed", right: 20, bottom: 20, width: 240, padding: 16,
          background: "white", border: "1px solid var(--line)", borderRadius: 10,
          boxShadow: "0 10px 30px oklch(0 0 0 / 0.1)", zIndex: 100,
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <span style={{ fontWeight: 600, fontSize: 13 }}>Tweaks</span>
            <button onClick={() => setTweaksOpen(false)} style={{ border: "none", background: "none", color: "var(--fg-mut)", fontSize: 18, lineHeight: 1 }}>×</button>
          </div>
          <div style={{ fontSize: 11, color: "var(--fg-mut)", marginBottom: 6 }}>Accent</div>
          <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
            {[
              { l: "Clay", v: "0.56 0.16 35" },
              { l: "Indigo", v: "0.5 0.17 265" },
              { l: "Forest", v: "0.5 0.14 160" },
              { l: "Ink", v: "0.35 0.02 260" },
            ].map(c => (
              <button key={c.l} onClick={() => persist({ accent: c.v })}
                style={{
                  flex: 1, padding: "6px 4px", fontSize: 11,
                  border: `1px solid ${tweaks.accent === c.v ? `oklch(${c.v})` : "var(--line)"}`,
                  background: tweaks.accent === c.v ? `oklch(${c.v} / 0.1)` : "white",
                  color: `oklch(${c.v})`, borderRadius: 6,
                }}>{c.l}</button>
            ))}
          </div>
          <div style={{ fontSize: 11, color: "var(--fg-mut)", marginBottom: 6 }}>Intro banner</div>
          <button onClick={() => persist({ showHelp: !tweaks.showHelp })}
            style={{
              width: "100%", padding: 6, fontSize: 12,
              border: `1px solid ${tweaks.showHelp ? "var(--accent)" : "var(--line)"}`,
              background: tweaks.showHelp ? "var(--accent-soft)" : "white",
              color: tweaks.showHelp ? "var(--accent)" : "var(--fg)", borderRadius: 6,
            }}>{tweaks.showHelp ? "Visible" : "Hidden"}</button>
        </div>
      )}
    </div>
  );
}

Object.assign(window, { CastApp });
