from core.geminiclient import GeminiClient
from core.groqclient import GroqClient

print("=" * 70)
print("RUNNING GEMINI CLIENT")
print("=" * 70)
gemini_client = GeminiClient()
gemini_client.process_prompts()

print("\n\n")
print("=" * 70)
print("RUNNING GROQ CLIENT")
print("=" * 70)
groq_client = GroqClient()
groq_client.process_prompts()

print("\n\n")
print("=" * 70)
print("✓ BOTH MODELS COMPLETED!")
print("=" * 70)
print(f"Gemini outputs: output/gemini/")
print(f"Groq outputs: output/groq/")