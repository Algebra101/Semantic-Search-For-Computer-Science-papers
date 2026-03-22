class RetrievalEvaluator:
    @staticmethod
    def precision_at_k(relevance_list, k):
        top_k = relevance_list[:k]
        return sum(top_k) / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(relevance_list, total_relevant, k):
        top_k = relevance_list[:k]
        return sum(top_k) / total_relevant if total_relevant > 0 else 0.0

    @staticmethod
    def f1_score(precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def reciprocal_rank(relevance_list):
        for index, rel in enumerate(relevance_list, start=1):
            if rel == 1:
                return 1 / index
        return 0.0