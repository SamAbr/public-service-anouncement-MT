#!/usr/bin/env python3
"""
gen_brief_pdf.py - typeset PRESENTATION_BRIEF.md as a printable PDF.

Markdown -> HTML -> Chromium print. Not reportlab: this document is mostly
tables and running prose with a designed cover, and laying that out by drawing
on a canvas would be a lot of work for a worse result. Chromium gives real
typography, automatic pagination and live selectable text.

Design follows the banner and the deck so the three read as one project: USIU
navy #293D94 with gold #FFCA08 as the single accent, Poppins for headings,
Carlito for body (metric-compatible with Calibri, which the deck uses).

Output: PRESENTATION_BRIEF.pdf, A4 portrait.
"""
import base64
import pathlib
import re
import subprocess
import sys

import markdown

HERE = pathlib.Path(__file__).resolve().parent

# Takes a markdown file, so the same typesetting serves every document in docs/.
#   python gen_brief_pdf.py IMPLEMENTATION_REPORT.md "Implementation report" "subtitle"
STEM = sys.argv[1] if len(sys.argv) > 1 else "PRESENTATION_BRIEF.md"
COVER_TITLE = sys.argv[2] if len(sys.argv) > 2 else \
    "Project brief:<br>everything you need<br>to present this"
COVER_LEDE = sys.argv[3] if len(sys.argv) > 3 else \
    ("Fine-Tuning Neural Machine Translation for Kenyan Public Service "
     "Announcements, adding Ekegusii to NLLB-200.")
SRC = HERE / STEM
OUT = HERE / (pathlib.Path(STEM).stem + ".pdf")
LOGO = base64.b64encode((HERE / "usiu_logo.png").read_bytes()).decode()

NAVY, GOLD = "#293d94", "#ffca08"
INK, INK2, INK3 = "#16181d", "#4b5563", "#8a919e"
TINT, TINT_G, LINE = "#f1f3f9", "#fff7de", "#dfe3ec"

TEAM = ["Weldesenbet Zeray", "Samuel Abrha", "Hetal Kumbharana",
        "Halima Mohammed", "Peter Kidiga", "Mitchelle Moraa"]

raw = SRC.read_text(encoding="utf-8")

# The first heading and the two lines under it become the cover; the rest is the
# body. Splitting here keeps the markdown source readable on its own.
body_md = raw.split("---", 2)[-1] if raw.count("---") >= 2 else raw

html_body = markdown.markdown(
    body_md, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])

# Sections FLOW. An earlier version forced a page break before each one, which
# is the obvious thing to do and wrong: it left most pages two-thirds empty and
# turned an 8-page document into 13. Headings carry page-break-after: avoid
# instead, so a heading can never strand at the foot of a page without its
# content - which is the actual problem a forced break was solving.

TEAM_HTML = " &middot; ".join(TEAM)

HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  @page {{ size: A4; margin: 18mm 16mm 16mm 16mm; }}
  @page :first {{ margin: 0; }}

  * {{ box-sizing: border-box; }}
  body {{ font-family: Carlito, "DejaVu Sans", sans-serif; font-size: 10.4pt;
          line-height: 1.52; color: {INK}; margin: 0; }}

  /* ---- cover ---- */
  .cover {{ page-break-after: always; background: {NAVY}; color: #fff;
            width: 210mm; height: 297mm; padding: 26mm 22mm; display: flex;
            flex-direction: column; }}
  .cover .logo {{ background: #fff; padding: 7px 10px; border-radius: 4px;
                  align-self: flex-start; }}
  .cover .logo img {{ height: 44px; display: block; }}
  .cover h1 {{ font-family: Poppins, sans-serif; font-size: 31pt; line-height: 1.14;
               font-weight: 700; margin: 40mm 0 0; letter-spacing: -.02em;
               color: #fff; }}
  .cover .lede {{ font-size: 13pt; color: #c9d2f0; margin-top: 8mm;
                  line-height: 1.45; max-width: 135mm; }}
  .cover .rule {{ width: 34mm; height: 3px; background: {GOLD}; margin: 12mm 0 8mm; }}
  .cover .who {{ font-size: 10.5pt; color: #fff; font-weight: 600;
                 line-height: 1.6; max-width: 140mm; }}
  .cover .who span {{ display: block; color: #9aa8d8; font-weight: 400;
                      font-size: 9.5pt; letter-spacing: .08em;
                      text-transform: uppercase; margin-bottom: 2mm; }}
  .cover .foot {{ margin-top: auto; font-size: 9.5pt; color: #9aa8d8;
                  line-height: 1.6; }}

  /* ---- headings ---- */
  h1, h2, h3 {{ font-family: Poppins, sans-serif; color: {NAVY};
                letter-spacing: -.01em; }}
  h2 {{ font-size: 16pt; font-weight: 700; margin: 10mm 0 5mm;
        padding-bottom: 3mm; border-bottom: 2px solid {LINE}; }}
  h2:first-of-type {{ margin-top: 0; }}
  h3 {{ font-size: 11.5pt; font-weight: 600; color: {INK};
        margin: 7mm 0 2.5mm; }}
  p {{ margin: 0 0 3.4mm; }}
  strong {{ font-weight: 700; color: {INK}; }}

  /* ---- tables ---- */
  table {{ border-collapse: collapse; width: 100%; margin: 4mm 0 5mm;
           font-size: 9.3pt; page-break-inside: avoid; }}
  th {{ background: {TINT}; color: {NAVY}; font-weight: 700; text-align: left;
        padding: 2.4mm 3mm; border-bottom: 1.5px solid {LINE};
        font-size: 8.6pt; letter-spacing: .03em; text-transform: uppercase; }}
  td {{ padding: 2.4mm 3mm; border-bottom: 1px solid {LINE}; vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  th[align="right"], td[align="right"] {{ text-align: right; }}

  /* ---- blockquote: the one thing on the page that is a voice, not a fact ---- */
  blockquote {{ margin: 5mm 0; padding: 4mm 6mm; background: {TINT_G};
                border-radius: 3mm; font-size: 10.2pt; color: {INK}; }}
  blockquote p {{ margin: 0; font-style: italic; }}

  code {{ font-family: "DejaVu Sans Mono", monospace; font-size: 8.8pt;
          background: {TINT}; padding: 0.4mm 1.4mm; border-radius: 1mm;
          color: {NAVY}; }}
  ul, ol {{ margin: 0 0 3.4mm; padding-left: 6mm; }}
  li {{ margin-bottom: 1.6mm; }}
  hr {{ border: none; border-top: 1px solid {LINE}; margin: 7mm 0; }}
  a {{ color: {NAVY}; text-decoration: none; }}
  em {{ color: {INK2}; }}

  /* Never orphan a heading from what it introduces. */
  h2, h3 {{ page-break-after: avoid; }}
</style></head>
<body>

<div class="cover" id="cover">
  <div class="logo"><img src="data:image/png;base64,{LOGO}" alt="USIU-Africa"></div>
  <h1>{COVER_TITLE}</h1>
  <div class="lede">{COVER_LEDE}</div>
  <div class="rule"></div>
  <div class="who"><span>Team</span>{TEAM_HTML}</div>
  <div class="who" style="margin-top:6mm"><span>Supervisor</span>Prof. Edward Ombui</div>
  <div class="foot">United States International University–Africa<br>
    Natural Language Processing · School of Science and Technology · 2026</div>
</div>

{html_body}

</body></html>"""

(HERE / f"_{OUT.stem}.html").write_text(HTML, encoding="utf-8")
print(f"brief.html  {len(HTML):,} bytes")

RENDER = f"""
import asyncio
from playwright.async_api import async_playwright

HEADER = ('<div style="font-family:Carlito,sans-serif;font-size:7.5pt;'
          'color:#8a919e;width:100%;padding:0 16mm;">'
          '<span>Ekegusii NMT &middot; USIU-Africa</span></div>')
FOOTER = ('<div style="font-family:Carlito,sans-serif;font-size:7.5pt;'
          'color:#8a919e;width:100%;padding:0 16mm;text-align:right;">'
          '<span class="pageNumber"></span></div>')
BLANK = '<div></div>'

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()

        # Cover: no header, no footer, no margins. Chromium stamps the header
        # on every page with no per-page escape, and navy-on-navy is invisible
        # rather than absent - so the cover is rendered separately and merged.
        pg = await b.new_page()
        await pg.goto("file://{HERE}/_{OUT.stem}.html#cover", wait_until="networkidle")
        await pg.add_style_tag(content="body > *:not(.cover) {{ display: none !important; }}")
        await pg.pdf(path="{HERE}/_c.pdf", format="A4", print_background=True,
                     margin={{"top": "0", "bottom": "0", "left": "0", "right": "0"}})
        await pg.close()

        pg = await b.new_page()
        await pg.goto("file://{HERE}/_{OUT.stem}.html", wait_until="networkidle")
        await pg.add_style_tag(content=".cover {{ display: none !important; }}")
        await pg.pdf(path="{HERE}/_b.pdf", format="A4", print_background=True,
                     display_header_footer=True,
                     header_template=HEADER, footer_template=FOOTER,
                     margin={{"top": "18mm", "bottom": "16mm",
                              "left": "16mm", "right": "16mm"}})
        await b.close()

asyncio.run(main())
"""
subprocess.run([sys.executable, "-c", RENDER], check=True)

from pypdf import PdfWriter
w = PdfWriter()
for part in ("_c.pdf", "_b.pdf"):
    w.append(str(HERE / part))
w.write(str(OUT))
w.close()
for part in ("_c.pdf", "_b.pdf"):
    (HERE / part).unlink()

from pypdf import PdfReader
r = PdfReader(str(OUT))
print(f"{OUT.name}  {OUT.stat().st_size / 1024:.0f} KB  ·  {len(r.pages)} pages")
