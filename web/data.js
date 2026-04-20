// Cast — fake corpus for the demo frontend.
window.CAST_DATA = (function () {
  const VIDEO = {
    id: "0c4b1e8a-9f3d-4b3e-9f9e-3b1c7f1a0001",
    title: "The Cartographer's Apprentice",
    subtitle: "Episode 03 · The Northern Passage",
    duration_sec: 2724,
    ingested_at: "Apr 17, 2026",
    status: "Ready",
  };

  const ENTITIES = [
    { id: "e1", label: "Elen Varo",      role: "Apprentice",       hue: 28,  n_dets: 412, conf: 0.97 },
    { id: "e2", label: "Master Khoren",  role: "Cartographer",     hue: 215, n_dets: 298, conf: 0.96 },
    { id: "e3", label: "Ira Solun",      role: "Smuggler captain", hue: 340, n_dets: 167, conf: 0.91 },
    { id: "e4", label: "The Envoy",      role: "Ministry agent",   hue: 160, n_dets: 94,  conf: 0.83 },
    { id: "e5", label: "Unnamed sailor", role: null,               hue: 50,  n_dets: 41,  conf: 0.72 },
  ];

  const CLIPS = [
    { idx: 0, start: 0, end: 46,
      caption: "Pre-dawn harbor. Lanterns on masts. A young woman hauls a rolled chart down a pier.",
      transcript: "— Keep it dry, it's the only copy. — I know what it is, Elen.",
      present: ["e1", "e2"],
      states: {
        e1: { feeling: "anxious",    knows: "route exists",     loyalty: "Khoren",   goal: "deliver the chart" },
        e2: { feeling: "guarded",    knows: "route exists",     loyalty: "guild",    goal: "reach the ship" },
      }},
    { idx: 1, start: 46, end: 141,
      caption: "Below deck of a schooner. Oil lamp. The captain unrolls a chart across a table.",
      transcript: "— The northern passage. You drew this? — My master did. I transcribed it.",
      present: ["e1", "e3"],
      states: {
        e1: { feeling: "measured",   knows: "route exists",     loyalty: "Khoren",   goal: "brief the captain" },
        e3: { feeling: "skeptical",  knows: "route exists",     loyalty: "crew",     goal: "assess the chart" },
      }},
    { idx: 2, start: 141, end: 268,
      caption: "Deck at sunrise. The crew sets sail. Elen watches the harbor recede.",
      transcript: "— We'll clear the breakwater by six bells. — And after that? — After that we trust your lines.",
      present: ["e1", "e3", "e5"],
      states: {
        e1: { feeling: "wistful",    knows: "route exists",     loyalty: "Khoren",   goal: "endure the passage" },
        e3: { feeling: "pragmatic",  knows: "route exists",     loyalty: "crew",     goal: "run the passage" },
      }},
    { idx: 3, start: 268, end: 402,
      caption: "Open sea. Fog. Ira studies the chart by lantern; a sailor marks the log.",
      transcript: "— The islands on your chart aren't on mine. — They were drawn from memory. — Whose memory?",
      present: ["e3", "e5"],
      states: {
        e3: { feeling: "uneasy",     knows: "chart is unusual", loyalty: "crew",     goal: "verify the chart" },
      }},
    { idx: 4, start: 402, end: 561,
      caption: "Night on deck. A signal lantern answers from the dark. A second vessel approaches.",
      transcript: "— That flag — I know that flag. Douse the lamps. — Too late. They saw us first.",
      present: ["e3", "e5"],
      states: {
        e3: { feeling: "alarmed",    knows: "pursuer present",  loyalty: "crew",     goal: "evade" },
      }},
    { idx: 5, start: 561, end: 712,
      caption: "A boarding action. Figures cross in silhouette. A woman in a long coat steps onto the deck.",
      transcript: "— Nobody moves. The chart, please. — You are very far from home, envoy.",
      present: ["e1", "e3", "e4"],
      states: {
        e1: { feeling: "afraid",     knows: "chart is hunted",  loyalty: "Khoren",   goal: "protect the chart" },
        e3: { feeling: "defiant",    knows: "pursuer present",  loyalty: "crew",     goal: "protect the crew" },
        e4: { feeling: "impassive",  knows: "chart exists",     loyalty: "ministry", goal: "seize the chart" },
      }},
    { idx: 6, start: 712, end: 889,
      caption: "Cabin. Elen alone with the envoy. A copper seal slid across the table.",
      transcript: "— You know what he made you carry, don't you? — I'm starting to.",
      present: ["e1", "e4"],
      states: {
        e1: { feeling: "destabilized", knows: "chart encodes more than a route", loyalty: "uncertain", goal: "understand" },
        e4: { feeling: "patient",      knows: "chart encodes more than a route", loyalty: "ministry",  goal: "turn the apprentice" },
      }},
    { idx: 7, start: 889, end: 1047,
      caption: "Flashback — a workshop. Khoren folds a chart into a book of psalms. Young Elen watches.",
      transcript: "— If they ever ask you what this is, it is nothing. It is a map of the coast.",
      present: ["e1", "e2"],
      states: {
        e1: { feeling: "dutiful",   knows: "chart is concealable", loyalty: "Khoren", goal: "obey" },
        e2: { feeling: "resolute",  knows: "chart is dangerous",   loyalty: "guild",  goal: "protect Elen" },
      }},
    { idx: 8, start: 1047, end: 1220,
      caption: "Back on the ship. Elen returns the seal to the envoy but palms a scrap of paper.",
      transcript: "— I'll consider your offer. — Consider quickly.",
      present: ["e1", "e4"],
      states: {
        e1: { feeling: "calculating", knows: "chart encodes more than a route", loyalty: "self",     goal: "buy time" },
        e4: { feeling: "suspicious",  knows: "chart encodes more than a route", loyalty: "ministry", goal: "observe" },
      }},
    { idx: 9, start: 1220, end: 1398,
      caption: "Galley. Ira and Elen, low voices. Elen slides the scrap across the bench.",
      transcript: "— This is half a latitude. — The other half is in my head. I won't say it on this ship.",
      present: ["e1", "e3"],
      states: {
        e1: { feeling: "resolute",    knows: "chart is leverage", loyalty: "self",  goal: "reach the passage" },
        e3: { feeling: "committed",   knows: "chart is leverage", loyalty: "Elen",  goal: "reach the passage" },
      }},
    { idx: 10, start: 1398, end: 1581,
      caption: "A storm. The schooner heels. Ira shouts. The envoy braces below.",
      transcript: "— Bear off, bear off! — Hold the line you were given!",
      present: ["e3", "e4", "e5"],
      states: {
        e3: { feeling: "overwhelmed", knows: "chart is leverage", loyalty: "Elen",     goal: "survive" },
        e4: { feeling: "anxious",     knows: "chart is leverage", loyalty: "ministry", goal: "survive" },
      }},
    { idx: 11, start: 1581, end: 1742,
      caption: "Calm after the storm. Dawn. An unfamiliar coastline. Elen at the bow.",
      transcript: "— That's not any coast I was taught. — No. It isn't.",
      present: ["e1", "e3"],
      states: {
        e1: { feeling: "awestruck",   knows: "passage is real",  loyalty: "self",  goal: "land" },
        e3: { feeling: "awestruck",   knows: "passage is real",  loyalty: "Elen",  goal: "land" },
      }},
    { idx: 12, start: 1742, end: 1903,
      caption: "Landfall. A cairn on a beach, glyphs carved into it. Elen touches them.",
      transcript: "— These are the same marks as the margins. — Then your master didn't draw this. He copied it.",
      present: ["e1", "e3"],
      states: {
        e1: { feeling: "grieving",    knows: "Khoren copied the chart", loyalty: "self", goal: "know the truth" },
      }},
    { idx: 13, start: 1903, end: 2081,
      caption: "The envoy confronts Elen on the beach. A pistol produced, then lowered.",
      transcript: "— You'll come back with me. — I'll come back. But not with you.",
      present: ["e1", "e4"],
      states: {
        e1: { feeling: "composed",     knows: "Khoren copied the chart", loyalty: "self",     goal: "return on her terms" },
        e4: { feeling: "recalculating",knows: "passage is real",         loyalty: "ministry", goal: "report" },
      }},
    { idx: 14, start: 2081, end: 2244,
      caption: "Ship under repair in a hidden cove. Ira hands Elen a journal.",
      transcript: "— It was in his sea chest. I didn't open it. — Then we open it together.",
      present: ["e1", "e3"],
      states: {
        e1: { feeling: "steady",      knows: "Khoren kept a journal", loyalty: "self", goal: "read the journal" },
      }},
    { idx: 15, start: 2244, end: 2412,
      caption: "Night cabin. Pages turned by lamplight. A folded chart falls out.",
      transcript: "— There's another one. — Of course there is.",
      present: ["e1", "e3"],
      states: {
        e1: { feeling: "resigned",    knows: "a second chart exists", loyalty: "self", goal: "plan next leg" },
      }},
    { idx: 16, start: 2412, end: 2556,
      caption: "Dawn. Schooner leaves the cove. The envoy watches from a cliff.",
      transcript: "— She's not coming back the way she left. — No, minister. She isn't.",
      present: ["e4"],
      states: {
        e4: { feeling: "chastened",   knows: "passage is real",  loyalty: "ministry", goal: "pursue" },
      }},
    { idx: 17, start: 2556, end: 2724,
      caption: "Elen at the wheel. Voice-over of Khoren's journal. End of episode.",
      transcript: "— \"If you are reading this, then the coast you were taught was a lie of mercy.\"",
      present: ["e1"],
      states: {
        e1: { feeling: "determined",  knows: "a second chart exists", loyalty: "self", goal: "find the next cartographer" },
      }},
  ];

  const EXAMPLE_QUERIES = [
    { label: "Someone realizes they've been lied to", text: "someone realizes they've been lied to", entity: null, predicate: null },
    { label: "Scenes with Elen and the Envoy",        text: "",                                    entity: "e1", predicate: null, entity2: "e4" },
    { label: "A chart is taken or handed over",        text: "chart taken handed",                  entity: null, predicate: null },
    { label: "Elen's loyalty shifts",                  text: "",                                    entity: "e1", predicate: "loyalty:shift" },
  ];

  return { VIDEO, ENTITIES, CLIPS, EXAMPLE_QUERIES };
})();
