import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

# load .env (expects GROQ_API_KEY inside src/core/.env)
load_dotenv()

class GroqClient:
    def __init__(self, config_path: str = None):
        """
        Initialize GroqClient with config from YAML file
        
        Args:
            config_path: Path to the YAML config file. If None, uses default path.
        """
        # Load config
        if config_path is None:
            # Default path relative to this file
            config_path = Path(__file__).parent.parent.parent / "config" / "groq_model_config.yaml"
        else:
            config_path = Path(config_path)
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        self.model_cfg = config["model"]
        self.paths_cfg = config["paths"]
        
        # Initialize the Groq client
        self.client = Groq()
        self.model_name = self.model_cfg["name"]
        self.temperature = self.model_cfg["temperature"]
        self.max_tokens = self.model_cfg["max_output_tokens"]
        self.top_p = self.model_cfg.get("top_p", 1)
        
        # Set up paths
        self.project_root = Path(__file__).parent.parent.parent
        self.prompts_root = self.project_root / self.paths_cfg["prompts_root"]
        self.output_root = self.project_root / self.paths_cfg["output_root"]
    
    def generate(self, prompt: str) -> str:
        """Generate response from a single prompt using streaming"""
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
    
    def save_pdf(self, text: str, path: Path):
        """Save text content to PDF file with improved formatting"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        doc = SimpleDocTemplate(
            str(path),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        styles = getSampleStyleSheet()
        story = []
        
        # Split text into lines and handle special characters
        lines = text.split("\n")
        
        for line in lines:
            if line.strip():  # Skip empty lines
                # Escape special characters for ReportLab
                line = line.replace('&', '&amp;')
                line = line.replace('<', '&lt;')
                line = line.replace('>', '&gt;')
                
                try:
                    para = Paragraph(line, styles["Normal"])
                    story.append(para)
                except Exception as e:
                    # If paragraph fails, use preformatted text
                    try:
                        para = Paragraph(f"<pre>{line}</pre>", styles["Code"])
                        story.append(para)
                    except:
                        # Last resort: skip problematic line
                        pass
            
            story.append(Spacer(1, 0.1*inch))
        
        try:
            doc.build(story)
        except Exception as e:
            print(f"\n⚠️  Error building PDF: {e}")
            # Fallback: save as plain text file
            txt_path = path.with_suffix('.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"   Saved as text file instead: {txt_path}")
    
    def process_prompts(self):
        """
        Read all text files from prompts directory,
        generate responses, and save to PDFs
        """
        print(f"🚀 Groq Client Starting...")
        print(f"Reading prompts from: {self.prompts_root}")
        print(f"Saving outputs to: {self.output_root}")
        print(f"Model: {self.model_name}")
        print("-" * 60)
        
        # Process all question folders (Q1, Q2, etc.)
        for q_folder in sorted(self.prompts_root.iterdir()):
            if not q_folder.is_dir():
                continue
            
            q_name = q_folder.name  # Q1, Q2 etc
            output_q_dir = self.output_root / q_name
            
            prompt_files = sorted(q_folder.glob("*.txt"))
            
            if not prompt_files:
                print(f"⚠️  No .txt files found in {q_name}")
                continue
            
            print(f"\n📁 Processing {q_name} ({len(prompt_files)} files)...")
            
            for i, prompt_file in enumerate(prompt_files, start=1):
                # Read the prompt
                with open(prompt_file, "r", encoding="utf-8") as f:
                    prompt_text = f.read()
                
                print(f"  [{i}/{len(prompt_files)}] Processing {prompt_file.name}...", end=" ")
                
                try:
                    # Generate response
                    response_text = self.generate(prompt_text)
                    
                    # Save to PDF
                    pdf_path = output_q_dir / f"{prompt_file.stem}.pdf"
                    self.save_pdf(response_text, pdf_path)
                    
                    print(f"✓ Saved to {pdf_path.relative_to(self.project_root)}")
                    
                except Exception as e:
                    print(f"✗ Error: {e}")
        
        print("\n" + "=" * 60)
        print("✓ All prompts processed!")


# Main execution
if __name__ == "__main__":
    # Create client and process all prompts
    client = GroqClient()
    client.process_prompts()