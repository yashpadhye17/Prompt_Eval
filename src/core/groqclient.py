import yaml
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# load .env (expects GROQ_API_KEY inside src/core/.env)
load_dotenv()


class GroqClient:
    def __init__(self, model_name: str, temperature: float, max_tokens: int, top_p: float):
        self.client = Groq()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def generate(self, prompt: str) -> str:
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            top_p=self.top_p,
            stream=True,
        )

        collected = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            collected.append(delta)

        return "".join(collected)


# -------------------------
# PDF helper
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
# Runner (same behavior as Gemini)
# -------------------------
def run_groq_pipeline():
    CONFIG_PATH = Path("../../config/model_config.yaml")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    paths_cfg = config["paths"]

    PROMPTS_ROOT = Path("../../") / paths_cfg["prompts_root"]
    OUTPUT_ROOT = Path("../../") / paths_cfg["output_root"]

    groq_client = GroqClient(
        model_name=model_cfg["name"],
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_output_tokens"],
        top_p=model_cfg.get("top_p", 1),
    )

    for q_folder in sorted(PROMPTS_ROOT.iterdir()):
        if not q_folder.is_dir():
            continue

        q_name = q_folder.name
        output_q_dir = OUTPUT_ROOT / q_name

        prompt_files = sorted(q_folder.glob("*.txt"))

        for prompt_file in prompt_files:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_text = f.read()

            response_text = groq_client.generate(prompt_text)

            pdf_path = output_q_dir / f"{prompt_file.stem}_groq.pdf"
            save_pdf(response_text, pdf_path)

            print(f"Saved → {pdf_path}")


if __name__ == "__main__":
    run_groq_pipeline()
