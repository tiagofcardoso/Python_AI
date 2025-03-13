from pathlib import Path
from langflow.base.data.utils import parse_text_file_to_data
from langchain_community.document_loaders import (
    PyMuPDFLoader,)
from langflow.custom import Component
from langflow.io import BoolInput, FileInput, Output
from langflow.schema import Data

class PDFComponent(Component):
    display_name = \"PDF\"\n    description = \"A PDF file loader.\"\n    icon = \"file-text\"\n\n    inputs = [\n        FileInput(\n            name=\"path\",\n            display_name=\"Path\",\n            file_types=[\"pdf\"],\n            info=f\"Supported file types: pdf\",\n        ),\n        BoolInput(\n            name=\"silent_errors\",\n            display_name=\"Silent Errors\",\n            advanced=True,\n            info=\"If true, errors will not raise an exception.\",\n        ),\n    ]\n\n    outputs = [\n        Output(display_name=\"Data\", name=\"data\", method=\"load_file\"),\n    ]\n\n    def load_file(self) -> Data:\n        if not self.path:\n            raise ValueError(\"Please, upload a PDF to use this component.\")\n        resolved_path = self.resolve_path(self.path)\n        silent_errors = self.silent_errors\n\n        extension = Path(resolved_path).suffix[1:].lower()\n\n        if extension != \"pdf\":\n            raise ValueError(f\"Unsupported file type: {extension}\")\n\n        loader = PyMuPDFLoader(str(self.path))\n        text = \" \".join(document.page_content for document in loader.load())\n        data = Data(data={\"file_path\": str(self.path), \"text\": text})\n        self.status = data if data else \"No data\"\n        return data or Data()\n