from dspy import BootstrapFewShot, Example
import dspy
from agent_poc.nodes.query_understanding import QueryUnderstanding


from agent_poc.semantic_layer.runtime import build_semantic_layer
from agent_poc.utils.dspy_helper import DspyHelper

DspyHelper.init()

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



# 2. 准备训练集和测试集
# Training set: used for optimization and generating few-shot demos (20 examples)
trainset = [
    Example(
        query="How many terminals are there in Delhi?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "City", "value": "Delhi"},
            {"type": "Facility", "value": "terminals"},
        ],
        intent="facility_count"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="How many terminals are there in Shanghai? List their code in a single line.",
        ontology_entities=ontology_list,
        entities=[
            {"type": "City", "value": "Shanghai"},
            {"type": "Facility", "value": "terminals"},
        ],
        intent="facility_list"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="How many containers were gated out of Sydney terminal on 10July2025?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "City", "value": "Sydney"},
            {"type": "Facility", "value": "terminal"},
            {"type": "ContainerEvent", "value": "gated out"},
        ],
        intent="container_event_count"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="Where is my container HASU1533926 on 20July2025?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Container", "value": "HASU1533926"},
        ],
        intent="container_tracking"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="What is the maximum weight load of a 40 DRY?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Container", "value": "40 DRY"},
        ],
        intent="container_specification"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="What is the total number of containers gated-in to Rotterdam last week?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "City", "value": "Rotterdam"},
            {"type": "ContainerEvent", "value": "gated-in"},
        ],
        intent="container_event_count"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="What is the height of container HASU1533926?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Container", "value": "HASU1533926"},
        ],
        intent="container_specification"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="Show me the details of shipment 256756918?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Shipment", "value": "256756918"}
        ],
        intent="shipment_details"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="How many units are linked to booking 250733952?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Shipment", "value": "250733952"}
        ],
        intent="shipment_container_count"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="Can you tell me how many container units are linked to shipment 256036146 please",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Shipment", "value": "256036146"}
        ],
        intent="shipment_container_count"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="Where is my container HASU1533926",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Container", "value": "HASU1533926"},
        ],
        intent="container_tracking"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="Can you tell me the current location of container HASU1533926?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Container", "value": "HASU1533926"},
        ],
        intent="container_tracking"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="What's the latest movement or location update for container HASU1533926?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Container", "value": "HASU1533926"},
        ],
        intent="container_tracking"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="What is the arrival date for my container GAOU7345026?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Container", "value": "GAOU7345026"},
        ],
        intent="container_eta"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="When can I expect my container GAOU7345026 to arrive?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Container", "value": "GAOU7345026"},
        ],
        intent="container_eta"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="What time will my container GAOU7345026 arrive?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Container", "value": "GAOU7345026"},
        ],
        intent="container_eta"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="show me the list of container linked to booking 250733952 in a tabular format",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Shipment", "value": "250733952"}
        ],
        intent="shipment_container_list"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="How many containers were gated out of Sydney terminal today?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "City", "value": "Sydney"},
            {"type": "Facility", "value": "terminal"},
            {"type": "ContainerEvent", "value": "gated out"},
        ],
        intent="container_event_count"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="What is the number of containers gated-in now in Rotterdam?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "City", "value": "Rotterdam"},
            {"type": "ContainerEvent", "value": "gated-in"},
        ],
        intent="container_event_count"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="How many terminals are there in Rotterdam? List their code in a single line.",
        ontology_entities=ontology_list,
        entities=[
            {"type": "City", "value": "Rotterdam"},
            {"type": "Facility", "value": "terminals"},
        ],
        intent="facility_list"
    ).with_inputs("query", "ontology_entities"),
]

# Test set: completely separate examples for final evaluation (10 examples)
testset = [
    Example(
        query="How many terminals are in Singapore?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "City", "value": "Singapore"},
            {"type": "Facility", "value": "terminals"},
        ],
        intent="facility_count"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="Where is container TEMU9876543 located?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Container", "value": "TEMU9876543"},
        ],
        intent="container_tracking"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="Show me shipment details for booking 987654321",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Shipment", "value": "987654321"}
        ],
        intent="shipment_details"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="How many containers are linked to booking 111222333?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Shipment", "value": "111222333"}
        ],
        intent="shipment_container_count"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="What is the weight capacity of a 20 REEF container?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Container", "value": "20 REEF"},
        ],
        intent="container_specification"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="When will container ABCD1234567 arrive?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Container", "value": "ABCD1234567"},
        ],
        intent="container_eta"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="How many containers were discharged at Hamburg port yesterday?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "City", "value": "Hamburg"},
            {"type": "Facility", "value": "port"},
            {"type": "ContainerEvent", "value": "discharged"},
        ],
        intent="container_event_count"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="List all terminals in Los Angeles",
        ontology_entities=ontology_list,
        entities=[
            {"type": "City", "value": "Los Angeles"},
            {"type": "Facility", "value": "terminals"},
        ],
        intent="facility_list"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="Display containers for shipment 555666777 in table format",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Shipment", "value": "555666777"}
        ],
        intent="shipment_container_list"
    ).with_inputs("query", "ontology_entities"),
    Example(
        query="What's the current position of container WXYZ9999999?",
        ontology_entities=ontology_list,
        entities=[
            {"type": "Container", "value": "WXYZ9999999"},
        ],
        intent="container_tracking"
    ).with_inputs("query", "ontology_entities"),
]


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
        expected_entities = {(e['type'], e['value']) for e in example.entities}
        predicted_entities = {(e['type'], e['value']) for e in pred.entities if isinstance(e, dict) and 'type' in e and 'value' in e}
        
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
OPTIMIZED_MODEL_PATH = "src/agent_poc/nodes/query_understanding_optimized.json"

# Check if optimized model already exists
model_path = Path(OPTIMIZED_MODEL_PATH)
if model_path.exists():
    print(f"Loading existing optimized model from {OPTIMIZED_MODEL_PATH}")
    compiled_step1 = QueryUnderstanding(semantic_layer)
    compiled_step1.load(OPTIMIZED_MODEL_PATH)
    print("✓ Optimized model loaded successfully")
else:
    print("No existing optimized model found. Running optimization...")
    # Compile step1 with examples and metric
    optimizer = BootstrapFewShot(
        metric=query_understanding_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=8
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
