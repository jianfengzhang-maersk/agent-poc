from dspy import MIPROv2, Example
import dspy
import yaml
from pathlib import Path
from agent_poc.modules.query_understanding.query_understanding import QueryUnderstanding


from agent_poc.semantic_layer.engine import build_semantic_layer
from agent_poc.utils.dspy_helper import DspyHelper

DspyHelper.init_kimi()

from agent_poc.semantic_layer.engine import build_semantic_layer, ontology_entities

# 1. Build the Step 1 module
step1 = QueryUnderstanding()


# 2. Load training and test data from YAML files
def load_examples_from_yaml(yaml_path: str) -> list:
    """Load examples from YAML file and convert to DSPy Example objects."""
    path = Path(yaml_path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    examples = []
    for item in data["examples"]:
        example = Example(
            query=item["query"],
            ontology_entities=ontology_entities,
            entities=item["entities"],
            intent=item["intent"],
        ).with_inputs("query", "ontology_entities")
        examples.append(example)

    return examples


# Load training set and test set from YAML files (in current directory)
trainset = load_examples_from_yaml(
    "src/agent_poc/modules/query_understanding/train_set.yaml"
)
testset = load_examples_from_yaml(
    "src/agent_poc/modules/query_understanding/test_set.yaml"
)

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
    if not pred or not hasattr(pred, "entities") or not hasattr(pred, "intent"):
        return 0.0

    # 1. Intent accuracy using LLM judge with CoT (50% of score)
    try:
        judge = dspy.ChainOfThought(IntentJudge)
        judgment = judge(expected_intent=example.intent, predicted_intent=pred.intent)
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
        expected_entities = {e for e in example.entities}
        predicted_entities = {e for e in pred.entities}

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


# 4. Define training and evaluation functions
from dspy.teleprompt import BootstrapFewShot
from pathlib import Path

# Path to save/load the optimized module
OPTIMIZED_MODEL_PATH = (
    "src/agent_poc/modules/query_understanding/query_understanding_optimized.json"
)


def evaluate_model(model, testset, metric_fn, title="Evaluating Model"):
    """
    Evaluate a model on test set and return metrics.

    Args:
        model: The DSPy module to evaluate
        testset: List of test examples
        metric_fn: Metric function to score predictions
        title: Title for the evaluation output

    Returns:
        dict: Evaluation results including average score and individual scores
    """
    print(f"\n{'='*80}")
    print(f"### {title} ###")
    print(f"{'='*80}")

    total_score = 0.0
    scores = []

    for i, example in enumerate(testset, 1):
        result = model(query=example.query, ontology_entities=example.ontology_entities)
        score = metric_fn(example, result)
        total_score += score
        scores.append(score)

        print(f"\n[Test {i}] Score: {score:.2f}")
        print(f"Query: {example.query}")
        print(f"Expected intent: {example.intent} | Predicted: {result.intent}")
        print(f"Expected entities: {example.entities} | Predicted: {result.entities}")

    avg_score = total_score / len(testset) if testset else 0
    print(f"\n{'='*80}")
    print(f"Average Score: {avg_score:.2f} ({avg_score*100:.1f}%)")
    print(f"{'='*80}\n")

    return {
        "average_score": avg_score,
        "scores": scores,
        "total_examples": len(testset),
    }


def train_model(base_model, trainset, metric_fn, save_path):
    """
    Train/optimize a model using MIPRO optimizer.

    Args:
        base_model: Base DSPy module to optimize
        trainset: Training examples
        metric_fn: Metric function for optimization
        save_path: Path to save the optimized model

    Returns:
        Compiled/optimized model
    """
    print("\n" + "=" * 80)
    print("### Starting Model Optimization ###")
    print("=" * 80)

    # MIPRO will optimize both instructions and demonstrations
    optimizer = MIPROv2(
        metric=metric_fn,
        auto="medium",  # Can be "light", "medium", or "heavy"
        verbose=True,
        track_stats=True,
    )

    compiled_model = optimizer.compile(base_model, trainset=trainset)

    # Save the optimized model
    compiled_model.save(save_path)
    print(f"\n✓ Optimized model saved to {save_path}")

    return compiled_model


def show_model_info(model):
    """Display information about the optimized model."""
    print("\n" + "=" * 80)
    print("### Model Information ###")
    print("=" * 80)

    for name, predictor in model.named_predictors():
        print(f"\n[Predictor: {name}]")
        print(f"Signature: {predictor.signature}")

        if hasattr(predictor, "demos") and predictor.demos:
            print(f"Number of demos: {len(predictor.demos)}")
        else:
            print("No demos found")

    print("\n" + "=" * 80)


if __name__ == "__main__":

    # 5. Main execution flow
    print("\n" + "=" * 80)
    print("### Query Understanding Optimization Pipeline ###")
    print("=" * 80)

    # Check if optimized model already exists
    model_path = Path(OPTIMIZED_MODEL_PATH)
    if model_path.exists():
        print(f"\nLoading existing optimized model from {OPTIMIZED_MODEL_PATH}")
        compiled_step1 = QueryUnderstanding()
        compiled_step1.load(OPTIMIZED_MODEL_PATH)
        print("✓ Optimized model loaded successfully")

        # Evaluate the loaded model
        results_loaded = evaluate_model(
            compiled_step1,
            testset,
            query_understanding_metric,
            "Evaluating Loaded Optimized Model",
        )

    else:
        print("\nNo existing optimized model found. Running optimization pipeline...")

        # Step 1: Evaluate baseline model (before optimization)
        print("\n>>> STEP 1: Baseline Evaluation (Before Optimization)")
        baseline_results = evaluate_model(
            step1, testset, query_understanding_metric, "Baseline Model Performance"
        )

        # Step 2: Train/optimize the model
        print("\n>>> STEP 2: Model Optimization")
        compiled_step1 = train_model(
            step1, trainset, query_understanding_metric, OPTIMIZED_MODEL_PATH
        )

        # Step 3: Evaluate optimized model (after optimization)
        print("\n>>> STEP 3: Post-Optimization Evaluation")
        optimized_results = evaluate_model(
            compiled_step1,
            testset,
            query_understanding_metric,
            "Optimized Model Performance",
        )

        # Step 4: Compare results
        print("\n" + "=" * 80)
        print("### Optimization Results Summary ###")
        print("=" * 80)
        print(
            f"Baseline Score:  {baseline_results['average_score']:.2f} ({baseline_results['average_score']*100:.1f}%)"
        )
        print(
            f"Optimized Score: {optimized_results['average_score']:.2f} ({optimized_results['average_score']*100:.1f}%)"
        )
        improvement = (
            optimized_results["average_score"] - baseline_results["average_score"]
        ) * 100
        print(f"Improvement:     {improvement:+.1f} percentage points")
        print("=" * 80)

    # Show model information
    show_model_info(compiled_step1)

    print("\n✓ Pipeline completed!")
