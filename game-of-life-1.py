from random import randint
import time
from rich.live import Live

def make_glider(n):
    if n < 3:
        raise Exception("n too small to make a glider")
    g = [[0 for _ in range(n)] for _ in range(n)]

    g[1][0] = 1
    g[2][1] = 1
    g[0][2] = 1
    g[1][2] = 1
    g[2][2] = 1

    return g 

def make_random(n):
    return [[randint(0, 1) for _ in range(n)] for _ in range(n)]

def count_neighbors(grid, x, y):
    total = 0
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if x + i <= len(grid)-1 and x + i >= 0 and \
               y + j <= len(grid[x])-1 and y + j >= 0:
                total += grid[x + i][y + j]
                
    total -= grid[x][y]
            
    return total


def draw(current):
    dsky = "  "
    for i, _ in enumerate(current):
        dsky += f"{i} "
    
    dsky += f"\tt: {t}\n"
    for y, row in enumerate(current):
        dsky += f"{y} "
        for x, col in enumerate(row):
            state = current[x][y]
            if state == 1: dsky += f"x "
            else: dsky += f"  "
        dsky += "\n"
    dsky += "\n"
    
    return dsky

# TODO: maybe use sparse storage, only keeping track of which ones are alive
# TODO: also probably use nmpy
def compute_next(current):
    # probably move this inside 
    grid_next = list(map(list, current))
    for y, row in enumerate(current):
        for x, col in enumerate(row):
            state = current[x][y]
            neighbors = count_neighbors(current, x, y)
            if state == 0 and neighbors == 3:
                grid_next[x][y] = 1
            elif state == 1 and neighbors < 2 or neighbors > 3:
                grid_next[x][y] = 0
            else: 
                grid_next[x][y] = state
            

    return grid_next

        

if __name__ == "__main__":
    n = 5
    grid = make_glider(n)
    t = 0
    epochs = 10
    f = open("output.txt", "w")
    with Live("", refresh_per_second=4) as live:
        while t < epochs:
            dsky = draw(grid)
            f.write(dsky)
            live.update(dsky)
            grid = compute_next(grid)
            time.sleep(0.25)
            t += 1

    f.close()