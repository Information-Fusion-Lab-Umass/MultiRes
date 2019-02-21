def lists_intersection(*lists):
    intersection = set()
    for idx, l in enumerate(lists):
        if idx == 0:
            intersection = set(l)
        else:
            intersection = intersection.intersection(set(l))

    return list(intersection)
