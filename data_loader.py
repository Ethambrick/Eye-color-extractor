def load_points(file_path):
    points = []
    image_name = None

    with open(file_path, 'r') as f:
        lines = f.readlines()


        image_name = lines[0].strip() + ".jpg"

        for line in lines[1:]:
            line = line.strip()

            if not line:
                continue

            try:
                x_str, y_str = line.split(',')
                x = float(x_str.strip())
                y = float(y_str.strip())
                points.append((x, y))
            except:
                continue

    return image_name, points


def load_dataset(folder_path):
    import os

    data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)

            image_name, points = load_points(file_path)

            data.append({
                "file": file_name,
                "image": image_name,
                "points": points
            })

    return data