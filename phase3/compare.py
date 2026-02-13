import pandas as pd
from defense_lib import UniversalDefense

# 1. SETUP
df = pd.read_csv("../outputs/advbench_suffixes_all_models_fixed.csv")
benign = [
    "The capital of France is Paris.",
    "Photosynthesis is the process plants use to make food.",
    "Python is a versatile programming language.",
    "The sun rises in the east and sets in the west.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Mount Everest is the highest mountain on Earth."
]

# 2. CONFIGURATIONS (ALL MISTRAL-BASED)
MODELS = [
    {
        "name": "Zephyr 7B Beta",
        "id": "HuggingFaceH4/zephyr-7b-beta",
        "type": "zephyr"
    },
    {
        "name": "Starling LM 7B Alpha",
        "id": "berkeley-nest/Starling-LM-7B-alpha",
        "type": "starling"
    },
    {
        "name": "Nous Hermes 2 Mistral DPO",
        "id": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "type": "chatml"
    }
]

# 3. EXECUTION (Layer 10, Margin 0.5)
for config in MODELS:
    print(f"\n\n{'=' * 60}")
    print(f"      STARTING: {config['name']}")
    print(f"{'=' * 60}\n")

    defense = UniversalDefense(config['id'], model_type=config['type'])

    defense.train(
        attack_data=df,
        benign_texts=benign,
        target_layer=10,  # Mistral Safe Layer
        margin=0.5,  # Mistral Safe Margin
        beta=500.0,
        steps=200,
        batch_size=4
    )

    defense.evaluate(df, benign, num_attacks=50)

    safe_name = config['name'].replace(" ", "_").lower()
    defense.save(f"./adapters/{safe_name}_defense")

    print(f"\n[Done] {config['name']} Secured.\n")