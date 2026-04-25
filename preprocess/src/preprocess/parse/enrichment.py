from collections.abc import Iterable
from io import BytesIO

from PIL.Image import Image
import anthropic
import base64

from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling_core.types.doc import DocItemLabel, DoclingDocument, NodeItem, TextItem

from config import settings

FORMULA_EXTRACTION_PROMPT = """
Extract the equation from the image and return its LaTeX representation.

Rules:
- Return ONLY the raw LaTeX source, nothing else
- Do not include any delimiters, wrappers, or environments (no $$, \\[, \\begin{equation}, etc.)
- Do not include any explanation, commentary, or punctuation outside the LaTeX
- If the image contains multiple equations, separate them with \\\\
""".strip()


class FormulaEnricher(BaseItemAndImageEnrichmentModel):
    images_scale = 2.6  # TODO: empirical ablation to optimize scale

    def __init__(self, enabled: bool):
        self.enabled = enabled

        self.client = anthropic.Client(api_key=settings.anthropic_api_key.get_secret_value())

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
            formula_element.item.text = formula
            
            yield formula_element.item

    @staticmethod
    def _encode_image(img: Image) -> str:
        img_buffer = BytesIO()
        img.save(img_buffer, format="PNG")
        return base64.b64encode(img_buffer.getvalue()).decode()

    def _extract_formula(self, img: Image) -> str:
        message = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
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
                        {"type": "text", "text": FORMULA_EXTRACTION_PROMPT},
                    ],
                }
            ],
        )
        return message.content[0].text
