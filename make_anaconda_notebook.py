"""Create a notebook copy for older SciPy permutation_test APIs."""
import json
from pathlib import Path

source_path = Path("AI_Dependency_Study_public.ipynb")
target_path = Path("AI_Dependency_Study_anaconda_compatible.ipynb")
notebook = json.loads(source_path.read_text(encoding="utf-8"))

changed = 0
for cell in notebook["cells"]:
    source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if "def seeded_permutation_test" in source:
        source = source.replace(
            'seed_argument = "rng" if "rng" in inspect.signature(stats.permutation_test).parameters else "random_state"',
            'seed_argument = "random_state"',
        ).replace(
            "**{seed_argument: np.random.default_rng(SEED)},",
            "**{seed_argument: SEED},",
        )
        cell["source"] = source
        changed += 1

for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

if changed != 1:
    raise RuntimeError(f"Expected to update one compatibility helper; updated {changed}.")

target_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Created {target_path.name}")
