from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent.resolve()  # src/esm2_mechanism/
PROJECT_ROOT = PACKAGE_ROOT.parent.parent  # esm2_mechanism/

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
PAPERS_DIR = PROJECT_ROOT / "papers"
DOCS_DIR = PROJECT_ROOT / "docs"
