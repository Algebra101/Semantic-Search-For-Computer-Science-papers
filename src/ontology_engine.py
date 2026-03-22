class OntologyExpander:
    def __init__(self):
        self.ontology_map = {
            "artificial intelligence": ["machine learning", "deep learning", "neural network"],
            "machine learning": ["classification", "prediction", "supervised learning"],
            "deep learning": ["cnn", "rnn", "transformer", "neural network"],
            "natural language processing": ["nlp", "text mining", "language model", "chatbot"],
            "information retrieval": ["semantic search", "document retrieval", "query expansion"],
            "cybersecurity": ["network security", "intrusion detection", "malware detection"],
            "cloud computing": ["virtualization", "distributed systems", "resource allocation"],
            "blockchain": ["smart contract", "distributed ledger", "web3"],
            "computer vision": ["image processing", "object detection", "image classification"],
            "internet of things": ["iot", "sensor network", "embedded system"],
            "database": ["sql", "dbms", "data storage"],
            "web development": ["frontend", "backend", "website", "web application"],
        }

    def expand_query_terms(self, query: str) -> list[str]:
        query = query.lower().strip()
        expanded = [query]

        for key, related_terms in self.ontology_map.items():
            if key in query or query in key:
                expanded.extend(related_terms)
            else:
                query_words = set(query.split())
                key_words = set(key.split())
                if query_words.intersection(key_words):
                    expanded.append(key)
                    expanded.extend(related_terms)

        seen = set()
        final_terms = []
        for term in expanded:
            if term not in seen:
                seen.add(term)
                final_terms.append(term)

        return final_terms

    def expand_query_text(self, query: str) -> str:
        return " ".join(self.expand_query_terms(query))