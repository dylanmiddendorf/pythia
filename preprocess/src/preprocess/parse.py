import re
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.document import DoclingDocument

_ANNOTATION_HEADERS = re.compile(
    r"^## \[Page \d+\]\s+(where|note:?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_ORDERED_LIST_HEADERS = re.compile(
    r"^## \[Page \d+\]\s+(\d+\. [a-z :]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions(do_formula_enrichment=True)
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # type: ignore

    # TODO: Improve LaTeX formula extraction (currently dogsh*t)
    # formula_engine = CodeFormulaVlmOptions.from_preset("codeformulav2")
    # pipeline_options.code_formula_options = formula_engine

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def extract_markdown(converter: DocumentConverter, pdf_path: str | Path) -> str:
    document = converter.convert(pdf_path).document
    _annotate_sections_with_page_numbers(document)

    md_text = document.export_to_markdown()
    md_text = _normalize_false_headers(md_text)

    return md_text


def _annotate_sections_with_page_numbers(document: DoclingDocument) -> None:
    for item in document.texts:
        if item.label == DocItemLabel.SECTION_HEADER and item.prov:
            item.text = f"[Page {item.prov[0].page_no}] {item.text}"


def _normalize_false_headers(md_text: str) -> str:
    md_text = _ANNOTATION_HEADERS.sub(r"\1\n", md_text)
    md_text = _ORDERED_LIST_HEADERS.sub(r"\1\n", md_text)
    return md_text

if __name__ == "__main__":
    conv = build_converter()
    with open("output.md", "wt") as output:
        output.write(extract_markdown(conv, "data/raw/TPS54331.pdf"))
