def sort_by_return(list_of_tuples):
    """
    sort list of tuples for a specific return template
    Args:
        list_of_tuples (list): list of 4 x,y tuples
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    list_of_tuples = sorted(list_of_tuples, key=lambda item: item[0])
    left_side = list_of_tuples[0:2]
    right_side = list_of_tuples[2:4]
    left_side = sorted(left_side, key=lambda item: item[1])
    right_side = sorted(right_side, key=lambda item: item[1])
    result = left_side + right_side
    return result


if __name__ == "__main__":
    list_of_tuples = [(0, 0), (3, 0), (1, 2), (2, 1)]
    print(sort_by_return(list_of_tuples))
