def extract_features(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    width = max_x - min_x
    height = max_y - min_y

    # normalize points
    norm_xs = [(x - min_x) / width if width != 0 else 0 for x in xs]
    norm_ys = [(y - min_y) / height if height != 0 else 0 for y in ys]

    avg_x = sum(norm_xs) / len(norm_xs)
    avg_y = sum(norm_ys) / len(norm_ys)

    density = len(points) / (width * height) if width * height != 0 else 0

    return [
        width,
        height,
        avg_x,
        avg_y,
        density
    ]