import base64
import io
import re
from abc import ABC, abstractmethod
from pathlib import Path

import anthropic
import pymupdf
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from pythia import settings
from pythia.parse.pipeline import DatasheetPipeline

_PROMPT = """
Transcribe this datasheet excerpt into Markdown faithfully. Do not invent, \
interpolate, or infer values not present in the source. When content is \
ambiguous, partially legible, or cut off at the excerpt boundary, transcribe \
what is clearly readable and stop.

Preserve all substantive content: part numbers, specs, conditions, footnotes, \
captions, warnings, and cross-references.

Drop page-level boilerplate: repeated running headers/footers (part-number \
stamps, "Submit Documentation Feedback", copyright/legal lines, bare page \
numbers, revision codes), and navigational sections that duplicate structure \
without adding content (Table of Contents, List of Figures, List of Tables).

**Revision history**: Preserve entries as a list. Strip leader dots and 
normalize page references to a parenthetical.

Source:  - Added the TPS62933 information to *Features* .......1
Output:  - Added the TPS62933 information to *Features* (p. 1)

## Page anchoring and headings

Pages start at page number {page} and increment. Embed the page number \
in every heading via an HTML comment after the `#` markers.

Heading depth maps strictly to section number depth: **(components separated \
by dots) + 1**, where the +1 reserves `#` for the document title.

- `#` — document title (the part number). Never use for sections.
- `##` — one component (`1`, `A`) or any unnumbered top-level section (`Features`, `Description`, `Package Outline`).
- `###` — two components (`1.2`, `A.1`), or a sub-heading under an unnumbered section.
- `####` — three components (`1.2.3`). Continue this pattern for deeper levels.

Examples:
- `# <!-- page: 1 --> TPS54331 3-A, 28-V Input, Step Down DC-DC Converter With Eco-mode`
- `## <!-- page: 1 --> 1 Features`
- `## <!-- page: 38 --> PACKAGE OUTLINE`
- `### <!-- page: 38 --> D0008A — SOIC - 1.75 mm max height`
- `##### <!-- page: 17 --> 8.2.2.6 Capacitor Selection`

Use section numbers exactly as printed; don't derive from the page number. If \
a section continues across pages, do not add a new heading with "(continued)" \
— just continue the content under the existing heading. If the excerpt begins \
mid-section with no visible heading (or the document title isn't in the \
excerpt), start with the first visible content; do not fabricate.

## Math

Favor semantic clarity over typographic precision; approximate rendering is \
fine as long as variable identities and relationships are preserved.

- Display equations: `$$ ... $$`. If labeled `(N)` on the right, append `\\tag{N}`.
- Inline equations and subscripted variable names: `$ ... $` (e.g., `$V_{CC}$`, `$I_{out}$`, `$f_{SW}$`).
- Plain values, units, ranges stay plain: `5 V`, `10 µF`, `-40 °C to 85 °C`, `±5%`.
- Active-low signals: `$\\overline{\\text{RESET}}$` in math; bare name (`RESET`) in prose and cells. Be consistent.

## Tables

Default to Markdown tables. Switch to HTML `<table>` only when the table requires:

- Merged cells
- Multi-row headers or grouped column headings (see the PIN / NO. + NAME pattern in the example below)
- Multi-line content within a cell

Universal rules (both formats):
1. Preserve the caption ("Table 5-1. Pin Functions") as plain text immediately above the table.
2. Keep footnote markers (`(1)`, `*`) inside cells; reproduce footnote text as a list below the table.
3. Tables spanning pages: render as one continuous table; do not duplicate the header row.

HTML-specific rules:
- Use `colspan`/`rowspan` for merged cells.
- Empty cells: `<td></td>`. No dashes or "N/A".

Example:

```html
<table>
  <thead>
    <tr><th colspan="2">PIN</th><th rowspan="2">I/O</th><th rowspan="2">DESCRIPTION</th></tr>
    <tr><th>NO.</th><th>NAME</th></tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>BOOT</td><td>O</td><td>A 0.1-μF bootstrap capacitor is...</td></tr>
  </tbody>
</table>
```

Pin diagrams (package outlines with labeled pins) are tabular — render as a \
two-column table of number and name, not as a figure.

## Visuals (schematics, block diagrams, plots, timing diagrams, waveforms)

Visuals are quantized to a brief description — enough signal for an embedding \
model to recognize what the figure depicts. **Do not** transcribe component \
values, axis values, curve data, or interconnections from inside a figure.

Format: `<!-- figure: <type> — <one-sentence description> -->`

`<type>` is one of: `plot`, `waveform`, `schematic`, `block-diagram`, \
`timing-diagram`, `layout-recommendation`, `mechanical-drawing`, `other`.

If the figure has an author-written caption, preserve it verbatim as plain \
text immediately below the comment:

```
<!-- figure: schematic — simplified application schematic for TPS54331D showing ... -->
Figure 8-3. Typical Application Circuit, 5 V Output
```

## Other Content

- **Notes, warnings, cautions** (boxed callouts): render as blockquotes with a leading bold label (`> **Note:** ...`, `> **Warning:** ...`).
- **Cross-references** ("see Section 7.3", "Figure 8-2", "Equation 4"): keep as plain prose; do not turn into links.
- **Multi-column layouts**: read top-to-bottom within each column, completing the left column before starting the right. Do not interleave columns.
"""


class AbstractDocumentConverter(ABC):
    @abstractmethod
    def convert(self, pdf_path: Path) -> str:
        """Convert a PDF document to markdown text."""


class DoclingConverter(AbstractDocumentConverter):
    _PROSE_HEADERS = re.compile(
        r"^## <!-- page: \d+ --> (where|notes?:?|therefore:?)$",
        re.IGNORECASE | re.MULTILINE,
    )

    _LIST_ITEM_HEADERS = re.compile(
        r"^## <!-- page: \d+ --> (\d+\. [a-z :]+)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    def __init__(self):
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_cls=DatasheetPipeline),
            }
        )

    def convert(self, pdf_path: Path) -> str:
        document = self._converter.convert(pdf_path).document
        self._annotate_sections_with_page_numbers(document)

        md_text = document.export_to_markdown()
        md_text = self._unescape_page_annotations(md_text)
        md_text = self._normalize_whitespace(md_text)
        md_text = self._demote_false_headers(md_text)

    @staticmethod
    def _annotate_sections_with_page_numbers(document: DoclingDocument) -> None:
        for item in document.texts:
            if item.label == DocItemLabel.SECTION_HEADER and item.prov:
                item.text = f"<!-- page: {item.prov[0].page_no} --> {item.text}"

    @staticmethod
    def _unescape_page_annotations(text: str) -> str:
        return re.sub(
            r"&lt;!-- page: (\d+) --&gt;",
            r"<!-- page: \1 -->",
            text,
        )

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"[ \t]{2,}", " ", text)

    @staticmethod
    def _demote_false_headers(text: str) -> str:
        text = DoclingConverter._PROSE_HEADERS.sub(r"\1\n", text)
        text = DoclingConverter._LIST_ITEM_HEADERS.sub(r"\1\n", text)
        return text


import base64
import io
from collections.abc import Iterator
from pathlib import Path
from string import Template

import anthropic
import pymupdf


class AnthropicConverter(AbstractDocumentConverter):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def convert(self, pdf_path: Path) -> str:
        return "\n\n".join(
            self.convert_chunk(chunk, start_page) for chunk, start_page in self._split_pdf(pdf_path, pages_per_chunk=8)
        )

    def convert_chunk(self, pdf_chunk: str, start_page: int) -> str:
        system_prompt = _PROMPT.replace("{page}", f"{start_page}")
        message = self._client.messages.create(
            model=self._model,
            max_tokens=8192,
            temperature=0.0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_chunk,
                            },
                        },
                    ],
                }
            ],
        )
        return message.content[0].text

    @staticmethod
    def _split_pdf(pdf_path: Path, pages_per_chunk: int = 8) -> Iterator[tuple[str, int]]:
        """
        Split a PDF into chunks of `pages_per_chunk` pages, yielding
        (base64_chunk, start_page) tuples. start_page is 1-indexed.
        """
        with pymupdf.open(pdf_path) as doc:
            total_pages = len(doc)

            for start in range(0, total_pages, pages_per_chunk):
                end = min(start + pages_per_chunk, total_pages)

                chunk_doc = pymupdf.open()
                chunk_doc.insert_pdf(doc, from_page=start, to_page=end - 1)

                buf = io.BytesIO()
                chunk_doc.save(buf)
                chunk_doc.close()

                chunk_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
                yield chunk_b64, start + 1  # 1-indexed page number
