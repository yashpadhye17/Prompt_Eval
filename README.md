# Prompt Evaluations

A Python-based project for evaluating and comparing prompts across different LLM providers including Meta's Llama models via Groq and OpenAI models.

## 📋 Project Overview

This project provides a framework for testing and evaluating prompts across multiple language models. It supports:
- **Groq API**: Meta Llama 3.1 models
- **OpenAI API**: OSS (Open Source Software) models

## 🗂️ Project Structure

```
Prompt_Evaluations/
├── config/
│   ├── groq_model_config....
│   └── openai_model_config....
├── output/
│   ├── gptoss/
│   │   ├── Q1
│   │   └── Q2
│   └── llama3.1/
│       ├── Q1
│       └── Q2
├── src/
│   ├── core/
│   │   ├── __pycache__/
│   │   ├── .env
│   │   ├── groqclient.py
│   │   └── openaiclient.py
│   └── prompts/
│       ├── Q1
│       └── Q2
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Groq API key
- OpenAI API key (if using OpenAI models)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Prompt_Evaluations
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the `src/core/` directory:
   ```bash
   touch src/core/.env
   ```

4. **Add your API keys**
   
   Open `src/core/.env` and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   > **Note**: Replace `your_groq_api_key_here` with your actual Groq API key. You can obtain a Groq API key from [Groq Cloud](https://console.groq.com/).

## 🎯 Usage

### Running Groq Client (Meta Llama Models)

The Groq client uses Meta's Llama 3.1 models for prompt evaluation:

```bash
python src/core/groqclient.py
```

This will:
- Load prompts from the `src/prompts/` directory
- Process them using Meta Llama 3.1 models via Groq
- Save results to `output/llama3.1/`

### Running OpenAI Client (OSS Models)

The OpenAI client uses open-source models for evaluation:

```bash
python src/core/openaiclient.py
```

This will:
- Load prompts from the `src/prompts/` directory
- Process them using OSS models
- Save results to `output/gptoss/`

## 📁 Input/Output Structure

### Prompts
- Store your evaluation prompts in `src/prompts/Q1` and `src/prompts/Q2`
- Add additional question directories as needed

### Results
- **Llama 3.1 results**: Saved in `output/llama3.1/Q1` and `output/llama3.1/Q2`
- **OSS model results**: Saved in `output/gptoss/Q1` and `output/gptoss/Q2`

## ⚙️ Configuration

Model configurations are stored in:
- `config/groq_model_config....`: Configuration for Groq/Llama models
- `config/openai_model_config....`: Configuration for OpenAI models

Edit these files to:
- Change model parameters (temperature, max tokens, etc.)
- Switch between different model versions
- Adjust evaluation settings

## 🔧 Dependencies

Install all required dependencies using:
```bash
pip install -r requirements.txt
```

Common dependencies include:
- `groq`: Groq API client
- `openai`: OpenAI API client
- `python-dotenv`: Environment variable management
- Additional dependencies as specified in `requirements.txt`

## 📊 Evaluation Workflow

1. **Prepare Prompts**: Add your evaluation prompts to the `src/prompts/` directory
2. **Configure Models**: Adjust settings in the config files
3. **Run Evaluations**: Execute either `groqclient.py` or `openaiclient.py`
4. **Review Results**: Check the `output/` directory for results
5. **Compare**: Analyze outputs from different models

## 🔒 Security Notes

- **Never commit your `.env` file** to version control
- The `.gitignore` file is configured to exclude sensitive files
- Keep your API keys secure and rotate them regularly
- Monitor your API usage to avoid unexpected charges

## 📝 Environment Variables

Required environment variables in `src/core/.env`:

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Your Groq API key for Llama models | Yes (for Groq) |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

See the [LICENSE](LICENSE) file for details.

## 📧 Support

For issues and questions:
- Open an issue in the repository
- Check existing issues for solutions
- Review the documentation

## 🎓 Additional Resources

- [Groq Documentation](https://console.groq.com/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Meta Llama Documentation](https://llama.meta.com/)

---

**Happy Prompting! 🚀**