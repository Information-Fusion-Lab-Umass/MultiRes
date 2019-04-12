def lists_intersection(*lists):
    intersection = set()
    for idx, l in enumerate(lists):
        if idx == 0:
            intersection = set(l)
        else:
            intersection = intersection.intersection(set(l))

    return list(intersection)


def list_difference(list_a, list_b):
    return list(set(list_a) - set(list_b))
