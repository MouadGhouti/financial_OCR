# Secret Side Quest (Python)

Minimal Python project with a LandingAI + Streamlit proof of concept.

## Setup

```bash
cd /Users/mghouti/Documents/secret-side-quest
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with your LandingAI API key:

```bash
LANDINGAI_API_KEY=your_api_key_here
```

## Run the minimal CLI entrypoint

```bash
source .venv/bin/activate
python src/main.py
```

## Run the Streamlit Financial Extractor PoC

```bash
source .venv/bin/activate
streamlit run src/app.py
```
