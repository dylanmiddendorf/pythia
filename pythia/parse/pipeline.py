import logging

from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

from pythia.parse.enrichment import FormulaEnricher


class DatasheetPipelineOptions(PdfPipelineOptions):
    do_formula_understanding: bool = True


class DatasheetPipeline(StandardPdfPipeline):
    def __init__(self, pipeline_options: DatasheetPipelineOptions):
        super().__init__(pipeline_options)

        if pipeline_options.table_structure_options.mode != TableFormerMode.ACCURATE:
            logging.warning(
                "DatasheetPipeline is configured with TableFormer mode '%s'. "
                "For best results with datasheets, use TableFormerMode.ACCURATE. "
                "Table extraction quality may be degraded.",
                pipeline_options.table_structure_options.mode.value,
            )

        self.enrichment_pipe = [
            FormulaEnricher(enabled=self.pipeline_options.do_formula_understanding),
        ]

        if self.pipeline_options.do_formula_understanding:
            self.keep_backend = True

    @classmethod
    def get_default_options(cls) -> DatasheetPipelineOptions:
        return DatasheetPipelineOptions()
