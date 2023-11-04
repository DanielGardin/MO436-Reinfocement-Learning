def discrete(position): 
    x, y = position

    return (int(x + 0.5), int(y + 0.5))

def manhattan_distance(source, target):
    x_s, y_s = source
    x_t, y_t = target

    return abs(x_s - x_t) + abs(y_s - y_t)