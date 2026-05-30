"""Render a NYT-style visual knockout bracket (circular flags + connector lines).

Centered final: the left half flows R32 → Final left-to-right, the right half
mirrors it. Built as self-contained HTML/CSS so it drops into a Streamlit
components.html() iframe. The tree is binary (every match has exactly two
feeders), which lets the connector lines be drawn exactly with CSS.
"""

# team name -> ISO code understood by flagcdn.com
ISO = {
    "Algeria": "dz", "Argentina": "ar", "Australia": "au", "Austria": "at",
    "Belgium": "be", "Brazil": "br", "Cabo Verde": "cv", "Canada": "ca",
    "Colombia": "co", "Croatia": "hr", "Curaçao": "cw", "Côte d'Ivoire": "ci",
    "Ecuador": "ec", "Egypt": "eg", "England": "gb-eng", "France": "fr",
    "Germany": "de", "Ghana": "gh", "Haiti": "ht", "Iran": "ir", "Japan": "jp",
    "Jordan": "jo", "Mexico": "mx", "Morocco": "ma", "Netherlands": "nl",
    "New Zealand": "nz", "Norway": "no", "Panama": "pa", "Paraguay": "py",
    "Portugal": "pt", "Qatar": "qa", "Saudi Arabia": "sa", "Scotland": "gb-sct",
    "Senegal": "sn", "South Africa": "za", "South Korea": "kr", "Spain": "es",
    "Switzerland": "ch", "Tunisia": "tn", "USA": "us", "Uruguay": "uy",
    "Uzbekistan": "uz",
}

# Bracket tree: match_id -> (feeder_home, feeder_away). Leaves (R32) absent.
FEEDS = {
    # Round of 16
    89: (73, 75), 90: (74, 77), 93: (83, 84), 94: (81, 82),
    91: (76, 78), 92: (79, 80), 95: (86, 88), 96: (85, 87),
    # Quarter-finals
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),
    # Semi-finals
    101: (97, 98), 102: (99, 100),
}
LEFT_ROOT, RIGHT_ROOT, FINAL = 101, 102, 104

# Round groupings for the stacked mobile view (match_id ranges are fixed).
ROUNDS = [
    ("Round of 32", list(range(73, 89))),
    ("Round of 16", list(range(89, 97))),
    ("Quarter-finals", [97, 98, 99, 100]),
    ("Semi-finals", [101, 102]),
    ("Final", [104]),
]


def _flag(team):
    code = ISO.get(team)
    if code:
        return (f'<img class="flag" src="https://flagcdn.com/w40/{code}.png" '
                f'alt="{team}" title="{team}">')
    # placeholder (unresolved playoff slot)
    abbr = "".join(w[0] for w in team.split()[:2]).upper()[:3]
    return f'<span class="flag ph" title="{team}">{abbr}</span>'


def _team_row(team, score, won):
    cls = "team won" if won else "team lost"
    sc = "" if score is None else f'<span class="score">{score}</span>'
    return (f'<div class="{cls}"><span class="name" title="{team}">{team}</span>'
            f'{_flag(team)}{sc}</div>')


def _match_box(mid, ko, side):
    """One match card: two team rows, winner highlighted."""
    if mid not in ko:
        body = (_team_row("TBD", None, False) + _team_row("TBD", None, False))
    else:
        h, a, hg, ag, ws, pens = ko[mid]
        hw, aw = (ws == "home"), (ws == "away")
        p = " p" if pens else ""
        body = (_team_row(h, f"{hg}{p if hw else ''}", hw)
                + _team_row(a, f"{ag}{p if aw else ''}", aw))
    return f'<div class="match {side}">{body}</div>'


def _node(mid, ko, side):
    """Recursive subtree. side='l' (children on left) or 'r' (mirror)."""
    feed = FEEDS.get(mid)
    box = f'<div class="match-wrap {side}">{_match_box(mid, ko, side)}</div>'
    if not feed:
        return box
    kids = f'<div class="children {side}">' + "".join(_node(c, ko, side) for c in feed) + "</div>"
    inner = (kids + box) if side == "l" else (box + kids)
    return f'<div class="node {side}">{inner}</div>'


def render(ko, height=620):
    """Return an HTML string for components.html()."""
    left = _node(LEFT_ROOT, ko, "l")
    right = _node(RIGHT_ROOT, ko, "r")
    final = _match_box(FINAL, ko, "f")
    champ = ""
    if FINAL in ko:
        h, a, hg, ag, ws, pens = ko[FINAL]
        w = h if ws == "home" else a
        champ = f'<div class="champ">🏆 {w}</div>'
    css = """
    <style>
      .wrap{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f5f1e8;
            overflow:hidden;padding:8px 4px;color:#01002e;}
      /* JS scales .bracket to fit; .bracket-fit reserves the *scaled* box so
         there's no overflow (hence no horizontal scrollbar) and no dead space. */
      .bracket-fit{}
      .bracket{display:flex;align-items:stretch;width:1100px;min-width:1100px;height:__H__px;}
      .half{display:flex;flex:1;}
      .center{display:flex;flex-direction:column;align-items:center;
              justify-content:center;padding:0 10px;min-width:150px;}
      .champ{margin-top:10px;font-weight:700;font-size:15px;color:#00bb7f;}
      .node{display:flex;align-items:stretch;flex:1;}
      .children{display:flex;flex-direction:column;justify-content:center;
                position:relative;flex:1;}
      /* vertical line joining a pair of children (exact: matches sit at 25%/75%) */
      .children.l::after,.children.r::after{content:"";position:absolute;
                top:25%;bottom:25%;border-top:0;}
      .children.l::after{right:0;border-right:2px solid #d0d5dd;}
      .children.r::after{left:0;border-left:2px solid #d0d5dd;}
      .match-wrap{display:flex;align-items:center;position:relative;padding:0 14px;}
      /* horizontal stub from each match toward its parent */
      .match-wrap.l::before{content:"";position:absolute;left:0;top:50%;
            width:14px;height:2px;background:#d0d5dd;}
      .match-wrap.r::before{content:"";position:absolute;right:0;top:50%;
            width:14px;height:2px;background:#d0d5dd;}
      /* the very first column (R32 leaves) has no incoming line on the outer side */
      .match{background:#fff;border:1px solid #e5e7eb;border-radius:8px;
             box-shadow:0 1px 2px rgba(0,0,0,.06);min-width:118px;overflow:hidden;}
      .match.f{min-width:140px;border-color:#ff9d00;box-shadow:0 2px 8px rgba(255,157,0,.30);}
      .team{display:flex;align-items:center;gap:6px;padding:4px 7px;font-size:12px;
            border-bottom:1px solid #f0f1f3;}
      .team:last-child{border-bottom:0;}
      .team .name{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
      .team.won{font-weight:700;color:#01002e;}
      .team.lost{color:#9aa1ab;}
      .team.lost .flag{filter:grayscale(.7);opacity:.6;}
      .flag{width:20px;height:20px;border-radius:50%;object-fit:cover;
            border:1px solid #e5e7eb;flex:none;}
      .flag.ph{display:inline-flex;align-items:center;justify-content:center;
            background:#eef0f3;color:#667;font-size:8px;font-weight:700;}
      .score{min-width:14px;text-align:right;font-variant-numeric:tabular-nums;}
      /* ---- stacked mobile view (hidden on wide screens) ---- */
      .rounds{display:none;padding:2px 6px 8px;}
      .rnd{margin-bottom:16px;}
      .rnd-title{margin:0 0 8px;padding-bottom:4px;font-size:13px;font-weight:700;
            text-transform:uppercase;letter-spacing:.04em;color:#475467;
            border-bottom:2px solid #eef0f3;}
      .rounds .match{min-width:0;width:100%;margin-bottom:8px;}
      .rounds .team{font-size:14px;padding:7px 10px;}
      .rounds .flag{width:22px;height:22px;}
      .rounds .champ{margin:0 0 14px;text-align:center;font-size:18px;}
      @media (max-width:760px){
        .wrap{overflow-x:hidden;}
        .bracket{display:none;}
        .rounds{display:block;}
      }
    </style>
    """.replace("__H__", str(height))
    # stacked view for narrow screens: champion first, then Final → R32
    rounds = ""
    for name, mids in reversed(ROUNDS):
        cards = "".join(_match_box(m, ko, "m") for m in mids if m in ko)
        if cards:
            rounds += f'<div class="rnd"><div class="rnd-title">{name}</div>{cards}</div>'
    rounds = f'<div class="rounds">{champ}{rounds}</div>'
    # Fit-to-width: scale the fixed-width bracket down so it always fits the
    # iframe (never up past natural size), and tell Streamlit the true height so
    # the iframe resizes with it — no horizontal scrollbar, no wasted space.
    script = """
    <script>
      function sendHeight(h){
        try{ window.parent.postMessage(
          {isStreamlitMessage:true, type:"streamlit:setFrameHeight", height:Math.ceil(h)}, "*"); }
        catch(e){}
      }
      function fitBracket(){
        var wrap=document.querySelector('.wrap'),
            fit=document.querySelector('.bracket-fit'),
            br=document.querySelector('.bracket');
        if(!wrap||!fit||!br) return;
        if(br.offsetParent!==null){                  // wide layout → scale to fit
          var natW=br.offsetWidth || 1100, natH=__H__,
              availW=Math.max(0, wrap.clientWidth-8);
          if(availW>0){
            var s=Math.min(1, availW/natW);
            br.style.transformOrigin='top left';
            br.style.transform='scale('+s+')';
            fit.style.width=(natW*s)+'px';
            fit.style.height=(natH*s)+'px';
            fit.style.margin='0 auto';
          }
        } else {                                     // stacked mobile → no scaling
          fit.style.width=''; fit.style.height=''; fit.style.margin='';
          br.style.transform='';
        }
        sendHeight(document.documentElement.scrollHeight+4);
      }
      // The iframe's final width is applied AFTER first paint and without firing a
      // window 'resize', so watch the document for size changes and re-fit then.
      if(window.ResizeObserver){
        try{ new ResizeObserver(fitBracket).observe(document.documentElement); }catch(e){}
      }
      window.addEventListener('resize', fitBracket);
      window.addEventListener('load', fitBracket);
      if(document.readyState==='loading'){
        document.addEventListener('DOMContentLoaded', fitBracket);
      } else { fitBracket(); }
      setTimeout(fitBracket, 60); setTimeout(fitBracket, 300);  // after fonts/flags settle
    </script>
    """.replace("__H__", str(height))
    html = ('<meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            + css + '<div class="wrap"><div class="bracket-fit"><div class="bracket">'
            f'<div class="half lh">{left}</div>'
            f'<div class="center"><div class="rlabel">FINAL</div>{final}{champ}</div>'
            f'<div class="half rh">{right}</div>'
            f'</div></div>{rounds}</div>' + script)
    return html
