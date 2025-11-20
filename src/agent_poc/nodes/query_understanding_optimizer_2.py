from dspy import MIPROv2, Example
import dspy
import yaml
from pathlib import Path
from agent_poc.nodes.query_understanding import QueryUnderstanding


from agent_poc.semantic_layer.runtime import build_semantic_layer
from agent_poc.utils.dspy_helper import DspyHelper

DspyHelper.init_kimi()

semantic_layer = build_semantic_layer(
    "src/agent_poc/semantic_layer/ontology.yaml",
    "src/agent_poc/semantic_layer/tools.yaml",
)
    
# Build entity descriptions from semantic layer
ontology_list = []
for name, ent in semantic_layer.entities.items():
    desc = ent.description or ""
    ontology_list.append(f"{name}: {desc}")
            
# 1. 构造你的 Step 1 模块
step1 = QueryUnderstanding(semantic_layer)


# 2. Load training and test data from YAML files
def load_examples_from_yaml(yaml_path: str) -> list:
    """Load examples from YAML file and convert to DSPy Example objects."""
    path = Path(yaml_path)
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    examples = []
    for item in data['examples']:
        example = Example(
            query=item['query'],
            ontology_entities=ontology_list,
            entities=item['entities'],
            intent=item['intent']
        ).with_inputs("query", "ontology_entities")
        examples.append(example)
    
    return examples


# Load training set and test set from YAML files (in current directory)
trainset = load_examples_from_yaml("src/agent_poc/nodes/training_set.yaml")
testset = load_examples_from_yaml("src/agent_poc/nodes/test_set.yaml")

print(f"Loaded {len(trainset)} training examples")
print(f"Loaded {len(testset)} test examples")


# 3. Define evaluation metric using LLM as judge
class IntentJudge(dspy.Signature):
    """Evaluate if two intents are semantically equivalent."""
    
    expected_intent: str = dspy.InputField(desc="The expected/ground truth intent")
    predicted_intent: str = dspy.InputField(desc="The predicted intent")
    score: float = dspy.OutputField(desc="Semantic similarity score between 0 and 1")


def query_understanding_metric(example, pred, trace=None):
    """
    Metric to evaluate query understanding quality using LLM as judge.
    Returns a score between 0 and 1.
    """
    score = 0.0
    
    # Check if prediction has required fields
    if not pred or not hasattr(pred, 'entities') or not hasattr(pred, 'intent'):
        return 0.0
    
    # 1. Intent accuracy using LLM judge with CoT (50% of score)
    try:
        judge = dspy.ChainOfThought(IntentJudge)
        judgment = judge(
            expected_intent=example.intent,
            predicted_intent=pred.intent
        )
        intent_score = float(judgment.score)
        # Clamp score between 0 and 1
        intent_score = max(0.0, min(1.0, intent_score))
        score += 0.5 * intent_score
    except Exception as e:
        # Fallback to exact match if LLM judge fails
        if pred.intent == example.intent:
            score += 0.5
    
    # 2. Entity extraction (50% of score)
    if len(example.entities) > 0:
        # Extract expected entity types and values
        # expected_entities = {(e['type'], e['value']) for e in example.entities}
        # predicted_entities = {(e['type'], e['value']) for e in pred.entities if isinstance(e, dict) and 'type' in e and 'value' in e}
        
        # only check entity name
        expected_entities = {e['type'] for e in example.entities}
        predicted_entities = {e['type'] for e in pred.entities if isinstance(e, dict) and 'type' in e and 'value' in e}
    
        # Calculate precision and recall
        if len(predicted_entities) > 0:
            correct = len(expected_entities & predicted_entities)
            precision = correct / len(predicted_entities)
            recall = correct / len(expected_entities)
            
            # F1 score for entities
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                score += 0.5 * f1
    
    return score


# 4. 运行 Optimizer
from dspy.teleprompt import BootstrapFewShot
from pathlib import Path

# Path to save/load the optimized module
OPTIMIZED_MODEL_PATH = "src/agent_poc/nodes/query_understanding_optimized_2.json"

# Check if optimized model already exists
model_path = Path(OPTIMIZED_MODEL_PATH)
if model_path.exists():
    print(f"Loading existing optimized model from {OPTIMIZED_MODEL_PATH}")
    compiled_step1 = QueryUnderstanding(semantic_layer)
    compiled_step1.load(OPTIMIZED_MODEL_PATH)
    print("✓ Optimized model loaded successfully")
else:
    print("No existing optimized model found. Running optimization...")
    # MIPRO will optimize both instructions and demonstrations
    optimizer = MIPROv2(
        metric=query_understanding_metric,
        auto="light",  # Can be "light", "medium", or "heavy"
        verbose=True,
        track_stats=True,
    )

    compiled_step1 = optimizer.compile(step1, trainset=trainset)
    
    # Save the optimized model
    compiled_step1.save(OPTIMIZED_MODEL_PATH)
    print(f"✓ Optimized model saved to {OPTIMIZED_MODEL_PATH}")


print("=" * 80)
print("Compiled module:", compiled_step1)
print("=" * 80)

# View the optimized prompts and demos
print("\n### Predictor Information ###")
for name, predictor in compiled_step1.named_predictors():
    print(f"\n[Predictor: {name}]")
    print(f"Signature: {predictor.signature}")
    
    # Check if demos exist
    if hasattr(predictor, 'demos') and predictor.demos:
        print(f"\nNumber of demos: {len(predictor.demos)}")
        print("\nDemos:")
        for i, demo in enumerate(predictor.demos, 1):
            print(f"\n--- Demo {i} ---")
            print(demo)
    else:
        print("No demos found")
    
    # Show the extended signature if available
    if hasattr(predictor, 'extended_signature'):
        print(f"\nExtended Signature: {predictor.extended_signature}")

print("\n" + "=" * 80)

# Evaluate on test set (unseen examples)
print("\n### Evaluating on Test Set ###")
total_score = 0.0
for i, example in enumerate(testset, 1):
    result = compiled_step1(query=example.query, ontology_entities=ontology_list)
    score = query_understanding_metric(example, result)
    total_score += score
    
    print(f"\n[Test {i}] Score: {score:.2f}")
    print(f"Query: {example.query}")
    print(f"Expected entities: {example.entities}")
    print(f"Predicted entities: {result.entities}")
    print(f"Expected intent: {example.intent}")
    print(f"Predicted intent: {result.intent}")

avg_score = total_score / len(testset) if testset else 0
print(f"\n{'='*80}")
print(f"Average Test Score: {avg_score:.2f} ({avg_score*100:.1f}%)")
print(f"{'='*80}")

# Inspect the last prompt sent to LLM (if using DSPy with history)
if hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'history'):
    print("\n### Last LLM Interaction ###")
    if dspy.settings.lm.history:
        last_prompt = dspy.settings.lm.history[-1]
        print("\nLast prompt:")
        print(last_prompt)

print("\n" + "=" * 80)
