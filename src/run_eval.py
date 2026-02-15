import yaml
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from core.geminiclient import GeminiClient

# -------------------------
# Load YAML config
# -------------------------
CONFIG_PATH = Path("../config/model_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

model_cfg = config["model"]
paths_cfg = config["paths"]

PROMPTS_ROOT = Path("../") / paths_cfg["prompts"]
OUTPUT_ROOT = Path("../") / paths_cfg["outputs"]

# -------------------------
# Init model
# -------------------------
client = GeminiClient(
    model_name=model_cfg["name"],
    temperature=model_cfg["temperature"],
    max_tokens=model_cfg["max_output_tokens"],
)

# -------------------------
# PDF writer
# -------------------------
def save_pdf(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(path))
    styles = getSampleStyleSheet()
    flow = []

    for line in text.split("\n"):
        flow.append(Paragraph(line, styles["Normal"]))
        flow.append(Spacer(1, 8))

    doc.build(flow)

# -------------------------
# Process all questions
# -------------------------
for q_folder in sorted(PROMPTS_ROOT.iterdir()):
    if not q_folder.is_dir():
        continue

    q_name = q_folder.name  # Q1, Q2 etc
    output_q_dir = OUTPUT_ROOT / q_name

    prompt_files = sorted(q_folder.glob("*.txt"))

    for i, prompt_file in enumerate(prompt_files, start=1):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()

        response_text = client.generate(prompt_text)

        pdf_path = output_q_dir / f"{prompt_file.stem}.pdf"
        save_pdf(response_text, pdf_path)

        print(f"Saved → {pdf_path}")
