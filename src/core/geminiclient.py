import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# load .env from this folder
load_dotenv()

class GeminiClient:
    def __init__(self, config_path: str = None):
        """
        Initialize GeminiClient with config from YAML file
        
        Args:
            config_path: Path to the YAML config file. If None, uses default path.
        """
        # Load config
        if config_path is None:
            # Default path relative to this file
            config_path = Path(__file__).parent.parent.parent / "config" / "gemini_model_config.yaml"
        else:
            config_path = Path(config_path)
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        self.model_cfg = config["model"]
        self.paths_cfg = config["paths"]
        
        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_cfg["name"],
            temperature=self.model_cfg["temperature"],
            max_output_tokens=self.model_cfg["max_output_tokens"],
        )
        
        # Set up paths
        self.project_root = Path(__file__).parent.parent.parent
        self.prompts_root = self.project_root / self.paths_cfg["prompts_root"]
        self.output_root = self.project_root / self.paths_cfg["output_root"]
    
    def generate(self, prompt: str) -> str:
        """Generate response from a single prompt"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Handle both string and list responses
        if isinstance(response.content, list):
            return "".join([
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in response.content
            ])
        else:
            return response.content
    
    def save_pdf(self, text: str, path: Path):
        """Save text content to PDF file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(str(path))
        styles = getSampleStyleSheet()
        flow = []
        
        for line in text.split("\n"):
            flow.append(Paragraph(line, styles["Normal"]))
            flow.append(Spacer(1, 8))
        
        doc.build(flow)
    
    def process_prompts(self):
        """
        Read all text files from prompts directory,
        generate responses, and save to PDFs
        """
        print(f"Reading prompts from: {self.prompts_root}")
        print(f"Saving outputs to: {self.output_root}")
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
                
                # Generate response
                response_text = self.generate(prompt_text)
                
                # Save to PDF
                pdf_path = output_q_dir / f"{prompt_file.stem}.pdf"
                self.save_pdf(response_text, pdf_path)
                
                print(f"✓ Saved to {pdf_path.relative_to(self.project_root)}")
        
        print("\n" + "=" * 60)
        print("✓ All prompts processed!")


# Main execution
if __name__ == "__main__":
    # Create client and process all prompts
    client = GeminiClient()
    client.process_prompts()