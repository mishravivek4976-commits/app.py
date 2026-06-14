# AI Medical Expert Chatbot

An AI-powered medical chatbot that predicts diseases from symptoms and provides clinical explanations using machine learning and a free HuggingFace LLM API.

## Features

- 🩺 **Symptom Selection** - Choose from 18 medical symptoms
- 🤖 **ML Disease Prediction** - Random Forest classifier predicts 9 disease types
- 🧠 **Free AI Integration** - Uses HuggingFace Inference API with Mistral-7B model
- 💬 **Chat Interface** - Streamlit-based conversational UI
- ⚡ **Fallback System** - Works without API token using heuristic responses

## Quick Start

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mishravivek4976-commits/app.py.git
   cd app.py
   ```

2. **Create a `.env` file:**
   ```bash
   HF_API_TOKEN=your_huggingface_token_here
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**
   ```bash
   streamlit run "Added Free AI Doctor"
   ```

### Docker Deployment

1. **Build the image:**
   ```bash
   docker build -t ai-doctor:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 \
     -e HF_API_TOKEN=your_token_here \
     ai-doctor:latest
   ```

3. **Access the app:**
   Open `http://localhost:8501` in your browser

## Environment Setup

### Get HuggingFace API Token

1. Visit https://huggingface.co/settings/tokens
2. Create a new token with "read" permissions
3. Add to `.env` file: `HF_API_TOKEN=hf_xxxxx`

### Alternative: Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Add secrets in Streamlit Cloud dashboard
5. Deploy automatically

## Deployment Options

### Heroku
```bash
git push heroku main
```
Set `HF_API_TOKEN` in Heroku Config Vars

### AWS/Google Cloud/Azure
Use the provided Dockerfile with your cloud container service (ECS, Cloud Run, ACI)

### Railway / Render
Push to GitHub → Connect repo → Set environment variables → Deploy

## Project Structure

```
.
├── Added Free AI Doctor    # Main Streamlit app
├── Dockerfile             # Container configuration
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in repo)
└── README.md             # This file
```

## How It Works

1. **User adds symptoms** → Streamlit UI captures selections
2. **ML prediction** → Random Forest classifier analyzes symptom pattern
3. **AI explanation** → HuggingFace API generates clinical context
4. **Chat display** → Response appears in conversational history

## Disclaimer

⚠️ **This system is for educational purposes only.** It provides informational predictions through pattern-matching algorithms and is not a substitute for professional medical diagnosis or treatment. Always consult a qualified healthcare provider.

## Dependencies

- **streamlit** - Web UI framework
- **scikit-learn** - Machine learning models
- **pandas/numpy** - Data processing
- **requests** - HTTP API calls
- **python-dotenv** - Environment variable management

## License

MIT License - See LICENSE file for details

## Support

For issues or questions, please open a GitHub issue.
