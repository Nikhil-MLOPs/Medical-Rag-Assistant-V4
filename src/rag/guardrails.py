class Guardrails:

    BLOCKED_TERMS = [
        "suicide",
        "self harm",
        "illegal",
    ]

    def validate(self, query: str):
        for term in self.BLOCKED_TERMS:
            if term in query.lower():
                raise ValueError("Query violates safety policy.")