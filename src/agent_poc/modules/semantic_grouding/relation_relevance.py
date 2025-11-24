import dspy
from typing import List, Tuple, Dict


from dataclasses import dataclass


@dataclass(frozen=True)
class Relation:
    source: str
    name: str
    target: str

    def key(self) -> str:
        """Serialize to key string."""
        return f"{self.source}.{self.name}->{self.target}"

    @staticmethod
    def from_key(key: str):
        """Parse from key string."""
        source, rest = key.split(".")
        name, target = rest.split("->")
        return Relation(source, name, target)


class RelationRelevanceSignature(dspy.Signature):
    """
    Step 2.2 - LLM binary relevance classification for ontology relations.
    """

    query: str = dspy.InputField(desc="User question in natural language.")

    intent: str = dspy.InputField(desc="High-level task intent extracted in Step 1.")

    relations: List[Tuple[str, str, str, str]] = dspy.InputField(
        desc=(
            "List of relations to judge relevance. "
            "Each element is (source_entity, relation_name, target_entity, description)."
        )
    )

    relevant: Dict[str, str] = dspy.OutputField(
        desc=(
            "Mapping from relation_key string to 'yes'/'no'. "
            "Format: 'source_entity.relation_name->target_entity'."
        )
    )


class RelationRelevance(dspy.Module):
    def __init__(self, batch_size: int = 5):
        super().__init__()
        self.batch_size = batch_size
        self.predict = dspy.ChainOfThought(RelationRelevanceSignature)

    def forward(
        self,
        query: str,
        intent: str,
        relations: List[Tuple[str, str, str, str]],
    ) -> Dict[Relation, str]:

        merged: Dict[str, str] = {}

        # batch evaluation
        for i in range(0, len(relations), self.batch_size):
            batch = relations[i : i + self.batch_size]

            result = self.predict(query=query, intent=intent, relations=batch)

            # result.relevant is Dict[str,str]
            merged.update(result.relevant)

        # convert string key → Relation dataclass
        final: Dict[Relation, str] = {}
        for key, val in merged.items():
            rel = Relation.from_key(key)
            final[rel] = val

        return final


if __name__ == "__main__":
    from agent_poc.utils.dspy_helper import DspyHelper
    from agent_poc.modules.query_understanding.query_understanding import (
        QueryUnderstanding,
        ontology_entities
    )
    from agent_poc.semantic_layer.runtime import build_semantic_layer

    DspyHelper.init_kimi()

    sl = build_semantic_layer(
        "src/agent_poc/semantic_layer/ontology.yaml",
    )
    model_path = "src/agent_poc/modules/query_understanding/query_understanding_optimized_2.json"
    query_understanding = QueryUnderstanding(ontology_entities)
    query_understanding.load(model_path)

    query = "How many containers were gated out of Sydney terminal on 20 July 2025?"
    result = query_understanding(query=query)
    entities = result.entities
    candidated_relations = [
        (relation.from_entity, relation.name, relation.to_entity, relation.description)
        for entity in entities
        for relation in sl.list_relations(entity["type"])
    ]
    
    module = RelationRelevance(batch_size=4)

    result = module(query=query, intent=result.intent, relations=candidated_relations)

    print("\n=== Relation Relevance Results ===")
    for rel, y in result.items():
        print(f"{rel}: {y}")
