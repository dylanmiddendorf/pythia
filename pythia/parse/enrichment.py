from collections.abc import Iterable
from io import BytesIO
import logging

from PIL.Image import Image
import anthropic
import base64

from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling_core.types.doc import DocItemLabel, DoclingDocument, NodeItem, TextItem

from pythia import settings

FORMULA_EXTRACTION_PROMPT = """
Extract the equation(s) from the image as MathJax-compatible LaTeX.

- Output raw LaTeX only: no commentary, no markdown fences, no $$ or \\[ \\] wrappers.
- Math-mode environments (pmatrix, cases, aligned, etc.) are allowed where needed.
- Wrap any prose inside the equation in \\text{...}.
- If the image contains no equation, output exactly: NO_FORMULA_FOUND
""".strip()


class FormulaEnricher(BaseItemAndImageEnrichmentModel):
    images_scale = 2.6  # TODO: empirical ablation to optimize scale

    def __init__(self, enabled: bool):
        api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""

        if enabled and not api_key:
            logging.warning("Anthropic API key not set; disabling formula enrichment.")
            enabled = False

        self.enabled = enabled
        self.client = anthropic.Client(api_key=api_key) if enabled else None

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled and isinstance(element, TextItem) and element.label == DocItemLabel.FORMULA

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        if not self.enabled:
            return

        for formula_element in element_batch:
            assert isinstance(formula_element.item, TextItem)

            formula = self._extract_formula(formula_element.image)
            if formula == "NO_FORMULA_FOUND":
                formula = ""  # Serializes as `<!-- formula-not-decoded -->`
            formula_element.item.text = formula

            yield formula_element.item

    @staticmethod
    def _encode_image(img: Image) -> str:
        img_buffer = BytesIO()
        img.save(img_buffer, format="PNG")
        return base64.standard_b64encode(img_buffer.getvalue()).decode("utf-8")

    def _extract_formula(self, img: Image) -> str:
        message = self.client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            system=FORMULA_EXTRACTION_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": self._encode_image(img),
                            },
                        },
                    ],
                }
            ],
        )
        return message.content[0].text
