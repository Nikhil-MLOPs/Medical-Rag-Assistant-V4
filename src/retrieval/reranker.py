from sentence_transformers import CrossEncoder
from src.utils.logging import setup_logging

logger = setup_logging("Reranker")


class Reranker:

    def __init__(self, model_name: str, boost_strength: float = 0.3):
        """
        Cross-encoder reranker with metadata-aware boosting.

        Parameters
        ----------
        model_name : str
            CrossEncoder model name
        boost_strength : float
            Strength of metadata boosts
        """

        logger.info("Initializing CrossEncoder Reranker")

        self.model = CrossEncoder(model_name)
        self.boost_strength = boost_strength

        # medical intent mapping
        self.intent_section_map = {
            "cause": ["cause", "causes", "why", "etiology"],
            "symptom": ["symptom", "symptoms", "sign"],
            "treatment": ["treatment", "therapy", "management"],
            "definition": ["definition", "what is", "overview"],
            "diagnosis": ["diagnosis", "test"],
            "complication": ["complication", "risk", "risks"],
            "prevention": ["prevent", "prevention", "avoid", "prevented"]
        }

    def _detect_query_intent(self, query: str):
        """
        Detect medical intent from query.
        Example:
            'How is diabetes caused?' -> cause
        """

        q = query.lower()

        for intent, keywords in self.intent_section_map.items():
            for word in keywords:
                if word in q:
                    return intent

        return None

    def _topic_boost(self, query: str, topic: str):
        """
        Boost if query explicitly mentions the topic.
        """

        if not topic:
            return 0.0

        query_lower = query.lower()
        topic_lower = topic.lower()

        if topic_lower in query_lower:
            return self.boost_strength * 2

        return 0.0

    def _section_boost(self, intent: str, section: str):
        """
        Boost if section matches detected intent.
        """

        if not intent or not section:
            return 0.0

        section = section.lower()

        if intent == "cause" and "cause" in section:
            return self.boost_strength

        if intent == "symptom" and "symptom" in section:
            return self.boost_strength

        if intent == "treatment" and "treatment" in section:
            return self.boost_strength

        if intent == "definition" and "definition" in section:
            return self.boost_strength

        if intent == "diagnosis" and "diagnosis" in section:
            return self.boost_strength

        if intent == "complication" and "complication" in section:
            return self.boost_strength
        
        if intent == "prevention" and "prevention" in section:
            return self.boost_strength

        return 0.0

    def rerank(self, query: str, results):
        """
        Rerank retrieved chunks using CrossEncoder + metadata boosts.
        """

        if not results:
            return []

        logger.info("Running reranker")

        # cross encoder scoring
        pairs = [(query, r.text) for r in results]
        scores = self.model.predict(pairs)

        intent = self._detect_query_intent(query)

        for result, score in zip(results, scores):

            base_score = float(score)

            topic = result.metadata.get("topic", "")
            section = result.metadata.get("section", "")

            topic_boost = self._topic_boost(query, topic)
            section_boost = self._section_boost(intent, section)

            final_score = base_score + topic_boost + section_boost

            result.score = final_score

        # sort descending
        results.sort(key=lambda x: x.score, reverse=True)

        return results