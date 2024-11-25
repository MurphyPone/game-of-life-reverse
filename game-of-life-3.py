from random import randint
import time
from rich.live import Live
import numpy as np

def make_glider(n):
    if n < 3:
        raise Exception("n too small to make a glider")
    g = np.arange(n*n).reshape((n,n))
    g.fill(0)

    g[0][1] = 1
    g[1][2] = 1
    g[2][0] = 1
    g[2][1] = 1
    g[2][2] = 1

    return g.astype(np.int8)

def make_random(n):
    return np.vectorize(lambda x: np.floor(x * 2))(np.random.rand(n, n)).astype(np.int8)

def count_neighbors(grid, x, y):
    total = 0
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if x + i <= len(grid)-1 and x + i >= 0 and \
               y + j <= len(grid[x])-1 and y + j >= 0:
                total += grid[x + i][y + j]
                
    total -= grid[x][y]
            
    return total


def compute_next_and_draw(current):
    dsky = "  "
    for i, _ in enumerate(current):
        dsky += f"{i} "
    dsky += f"\tt: {t}, alive: {current.astype(int).sum((0, 1))}\n"

    grid_next = current.copy()
    for x, col in enumerate(current):
        dsky += f"{x} "
        for y, row in enumerate(col):
            state = current[x][y]

            if state == 1: dsky += f"x "
            else: dsky += f"  "

            neighbors = count_neighbors(current, x, y)
            if state == 0 and neighbors == 3:
                grid_next[x][y] = 1
            elif state == 1 and neighbors < 2 or neighbors > 3:
                grid_next[x][y] = 0
            else: 
                grid_next[x][y] = state
        dsky += "\n"
    dsky += "\n"
        
    return grid_next, dsky


if __name__ == "__main__":
    n = 5
    grid = make_glider(n)
    t = 0
    epochs = 10
    f = open("output.txt", "w")
    with Live("", refresh_per_second=4) as live:
        while t < epochs:
            grid, dsky = compute_next_and_draw(grid)
            f.write(dsky)
            live.update(dsky)
            time.sleep(0.25)
            t += 1

    f.close()