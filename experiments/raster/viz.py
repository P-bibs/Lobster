
def plot_bitmap(bitmap):
    out = ""
    out += "X" * (bitmap.shape[1] + 2) + "\n"
    for row in bitmap:
        out += "X"
        for col in row:
            if col == 1:
                out += "█"
            else:
                out += " "
        out += "X"
        out += "\n"
    out += "X" * (bitmap.shape[1] + 2) + "\n"
    return out

def print_bitmap(bitmap):
    print(plot_bitmap(bitmap))

def plot(tuples, grid_size):
    coords = [[0 for x in range(grid_size)] for y in range(grid_size)]
    for t in tuples:
        coords[t[1]][t[2]] = t[0]

    print("X" * (grid_size + 2))
    for row in coords:
        print("X", end="")
        for col in row:
            if col == 1:
                print("█", end="")
            else:
                print(" ", end="")
        print("X")
    print("X" * (grid_size + 2))

