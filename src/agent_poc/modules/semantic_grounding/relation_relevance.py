import dspy
from typing import List, Tuple, Dict

from agent_poc.semantic_layer.ontology import RelationKey


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
    ) -> Dict[RelationKey, str]:

        merged: Dict[str, str] = {}

        # batch evaluation
        for i in range(0, len(relations), self.batch_size):
            batch = relations[i : i + self.batch_size]

            result = self.predict(query=query, intent=intent, relations=batch)

            # result.relevant is Dict[str,str]
            merged.update(result.relevant)

        # convert string key → RelationKey tuple for downstream use
        final: Dict[RelationKey, str] = {}
        for key, val in merged.items():
            source, rest = key.split(".")
            name, target = rest.split("->")
            final[(source, name, target)] = val

        return final


if __name__ == "__main__":
    from agent_poc.utils.dspy_helper import DspyHelper
    from agent_poc.modules.query_understanding.query_understanding import (
        QueryUnderstanding,
        ontology_entities
    )
    from agent_poc.semantic_layer.engine import (
        build_semantic_layer,
        ONTOLOGY_SOURCE_PATH,
    )

    DspyHelper.init_kimi()

    sl = build_semantic_layer(ONTOLOGY_SOURCE_PATH)
    model_path = "src/agent_poc/modules/query_understanding/query_understanding_optimized_2.json"
    query_understanding = QueryUnderstanding(ontology_entities)
    query_understanding.load(model_path)

    query = "How many containers were gated out of Sydney terminal on 20 July 2025?"
    qu_result = query_understanding(query=query)
    entities = qu_result.entities
    candidated_relations = [
        (relation.from_entity, relation.name, relation.to_entity, relation.description)
        for entity in entities
        for relation in sl.list_relations(entity["type"])
    ]
    
    relation_relevance = RelationRelevance(batch_size=4)

    relevance_result = relation_relevance(query=query, intent=qu_result.intent, relations=candidated_relations)

    print("\n=== Relation Relevance Results ===")
    for rel, y in relevance_result.items():
        print(f"{rel}: {y}")
