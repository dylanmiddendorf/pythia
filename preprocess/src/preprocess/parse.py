import re
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

_PROSE_HEADERS = re.compile(
    r"^## <!-- page: \d+ --> (where|notes?:?|therefore:?)$",
    re.IGNORECASE | re.MULTILINE,
)

_LIST_ITEM_HEADERS = re.compile(
    r"^## <!-- page: \d+ --> (\d+\. [a-z :]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # type: ignore

    # TODO: Improve table extraction:
    # - Export to HTML
    # - Outsource to Claude Haiku 4.5

    # TODO: Improve formula extraction (currently dogsh*t)
    # formula_engine = CodeFormulaVlmOptions.from_preset("codeformulav2")
    # pipeline_options.code_formula_options = formula_engine

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def extract_markdown(converter: DocumentConverter, pdf_path: str | Path) -> str:
    # TODO: Improve page annotation infrastructure

    document = converter.convert(pdf_path).document
    _annotate_sections_with_page_numbers(document)

    md_text = document.export_to_markdown()
    md_text = _unescape_page_annotations(md_text)
    md_text = _normalize_whitespace(md_text)
    md_text = _demote_false_headers(md_text)

    return md_text


def _annotate_sections_with_page_numbers(document: DoclingDocument) -> None:
    for item in document.texts:
        if item.label == DocItemLabel.SECTION_HEADER and item.prov:
            item.text = f"<!-- page: {item.prov[0].page_no} --> {item.text}"


def _unescape_page_annotations(text: str) -> str:
    return re.sub(
        r"&lt;!-- page: (\d+) --&gt;",
        r"<!-- page: \1 -->",
        text,
    )


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", text)


def _demote_false_headers(text: str) -> str:
    text = _PROSE_HEADERS.sub(r"\1\n", text)
    text = _LIST_ITEM_HEADERS.sub(r"\1\n", text)
    return text


if __name__ == "__main__":
    conv = build_converter()
    with open("output.md", "wt") as output:
        output.write(extract_markdown(conv, "data/raw/TPS54331.pdf"))
