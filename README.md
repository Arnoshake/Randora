# 🌍 Randora: Procedural World & Civilization Simulator

Randora is a fantasy world generation and civilization simulation project written in Python. It creates topographic terrain, distributes natural resources, generates civilizations, simulates city growth, and can even produce a GPT-narrated history of your world.

## Features

- Procedural terrain generation (altitude, temperature, resource layers)
- Civilization spawning and territory claiming
- City building with upgradable infrastructure and resource usage
- Fantasy-style historical narration using GPT API (optional)
- Visual map output using Matplotlib

## Performance Notice

⚠️ **Randora is an unoptimized simulation. Expect it to take time to run.**  
The civilization and city update loops are not yet optimized, and GPT-based narration requests further add to execution time. Performance improvements are planned, but for now, patience is required.

## API Cost & Disclaimer

> **Note:** GPT-4 API integration is currently **disabled** by default.

The GPT-based history generator uses a significant number of tokens per run (due to detailed prompts and multiple time steps), which results in **an API cost of roughly $1 per full simulation**.

As a broke college student, I can’t afford to keep this running every time — so API calls are off unless explicitly enabled.

If you want to use the narration feature:
- Get your own [OpenAI API key](https://platform.openai.com/account/api-keys)
- Enable the `simulate_civ_history` function manually in the code
- Use at your own financial discretion!

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/randora.git
   cd randora
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your `.env` file if using the GPT narration:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

5. Run the world generator:
   ```bash
   python WorldGen.py
   ```

## Configuration

You can tweak the following parameters inside `WorldGen.py`:
- `WORLD_SIZE` — map dimensions
- `WORLD_SEED` — RNG seed for repeatability
- Enable/disable GPT narration in `simulate_civ_history`

## Project Structure

```
randora/
|
├── WorldGen.py             # Main script
├── gpt_uitls.py            # Civilization history narration 
├── histories/              # Saved outputs (maps, logs, etc.)
└── README.md               # You're reading it!
```

## Future Plans

- Optimize simulation performance
- Add trading, warfare, and diplomacy between civilizations
- Web-based UI using Flask or React
- Better map visualization and in-browser interaction

## Inspiration

This project blends:
- Dungeons & Dragons-style fantasy worldbuilding  
- Realistic procedural generation methods  
- SimCity-style city simulation  
- A sprinkle of GPT magic

## Contact

Created by **Zachary West**  
[LinkedIn](https://www.linkedin.com/in/zacharywest2004/) • [GitHub](https://github.com/Arnoshake)
