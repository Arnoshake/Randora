# gpt_utils.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30  # seconds
)



def simulate_civ_history(world_name: str,civ_name: str, civ_history: list[str], total_years: int = 1000, step: int = 100):
    """
    Uses GPT to narrate a civilization's history in connected steps,
    and writes the result to a .txt file named after the civilization.
    """
    total_cost = 0.0
    total_tokens = 0

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a historical narrator chronicling the civilization '{civ_name}' in the world of '{world_name}'. "
                "Each summary you generate should build upon the previous. Keep tone consistent and narrative flowing."
                "Narration and facts of the world should be fantasy in style and substance. Emphasis on a Dungeons & Dragons style fantasy"
            )
        }
    ]

    output_lines = []  
    current_year = 0

    for i, entry in enumerate(civ_history):
        next_year = current_year + step
        user_prompt = (
            f"You are a fantasy historian chronicling the civilization of {civ_name} in the world of {world_name}. "
            f"Using the following data from years {current_year} to {next_year}, write a short but vivid summary of this period. "
            f"Focus on major developments, challenges, or achievements, not numeric values. Avoid repeating static details. "
            f"Use a high fantasy tone, as if writing a chronicle or lore entry.\n\n{entry}"

        )

        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            #REMOVING ALL RANDOMNESS TO ENSURE RESPONSE DETERMINED BY WORLD SEED
            temperature=0,  # no randomness
            top_p=1,        # no nucleus sampling
            frequency_penalty=0,
            presence_penalty=0
        )

        reply = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": reply})

        output_lines.append(f"=== Year {current_year} to {next_year} ===\n{reply}\n")
        # Track token usage
        usage = response.usage
        total_tokens += usage.total_tokens
        cost = (usage.prompt_tokens * 0.01 + usage.completion_tokens * 0.03) / 1000
        total_cost += cost
        current_year = next_year
    

    base_folder = "histories"
    world_folder = os.path.join(base_folder, world_name)
    os.makedirs(world_folder, exist_ok=True)

    # Create the file inside that folder
    filename = os.path.join(world_folder, f"{civ_name.lower().replace(' ', '_')}_history.txt")

    # Write the GPT summary
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"Saved GPT-generated history to: {filename}")
    return total_cost