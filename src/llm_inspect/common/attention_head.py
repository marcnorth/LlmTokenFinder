class AttentionHead:
    """
    Class representing an attention head in a transformer model.
    :param layer: The 0-indexed layer index of the attention head in the model.
    :param head: The 0-indexed head index of the attention head in the layer.
    """
    def __init__(self, layer: int, head: int):
        self.layer = layer
        self.head = head

    def __eq__(self, other):
        if not isinstance(other, AttentionHead):
            return NotImplemented
        return self.layer == other.layer and self.head == other.head

    def __hash__(self):
        return hash((self.layer, self.head))

    def __repr__(self):
        return f"{self.layer}.{self.head}"

    @staticmethod
    def intersection(heads: list[list["AttentionHead"]]) -> list["AttentionHead"]:
        """
        Find the intersection of a list of lists of AttentionHead objects.
        :param heads: A list of lists of AttentionHead objects.
        :return: A list of AttentionHead objects that are in all of the lists.
        """
        if not heads:
            return []
        intersection = set(heads[0])
        for head_list in heads[1:]:
            intersection &= set(head_list)
        return list(intersection)
