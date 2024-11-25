import numpy as np
import os
from z3 import Int, Or, And, Not, Solver, sat

class Board():
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self._board = np.zeros((width, height), dtype=int)
        self.t = 0
        self.forbidden = self.load_states()

    def count_neighbors(self, x, y):
        """counts the number of living cells surrounding the cell at `x`, `y`"""
        total = -self[x][y]
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (x + i <= self.width-1 and x + i >= 0) and \
                    (y + j <= self.height-1 and y + j >= 0):
                    total += self[x + i][y + j]
        
        return total

    def compute_next(self):
        """computes the next state for the current board"""
        nxt = np.array(self._board, dtype=int)
        for x in range(self.width):
            for y in range(self.height):
                state = self[x][y]
                neighbors = self.count_neighbors(x, y)
                if state == 0 and neighbors == 3:
                    nxt[x][y] = 1
                elif state == 1 and (neighbors < 2 or neighbors > 3):
                    nxt[x][y] = 0
                else:
                    nxt[x][y] = state

        return nxt
    
    def __all_previous(self, s, initial_terms):
        def block_term(s, m, t):
            s.add(t != m.eval(t))
        def fix_term(s, m, t):
            s.add(t == m.eval(t))
        def all_smt_rec(terms):
            if sat == s.check():
                m = s.model()
                yield m

                for i in range(len(terms)):
                    s.push()
                    block_term(s, m, terms[i])
                    for j in range(i):
                        fix_term(s, m, terms[j])
                    yield from all_smt_rec(terms[i:])
                    s.pop()

        yield from all_smt_rec(list(initial_terms))  

    def model_to_board(self, sol):
            target_model = sorted(list(filter(lambda v: "t_" in str(v), sol)), key=lambda v: str(v))
            target_vals = [sol[v].as_long() for v in target_model]
            return np.reshape(np.array(target_vals), (self.width, self.height)).transpose()

    def compute_previous(self):
        # an array of apt dimension containing symbolic variables for Z3 to compute a valid, previous board state
        target = np.array([[Int(f"t_{r}_{c}") for r in range(self.width)] for c in range(self.height)])

        # an "intermediate" array of similar dimension to contain the number of neighbors each cell is allowed to have per the transition rules
        neighbors = np.array([[Int(f"n_{r}_{c}") for r in range(self.width)] for c in range(self.height)])
        solver = Solver()

        for x in range(self.width):
            for y in range(self.height):
                curr_cell = self[x][y]
                target_cell = target[x][y]
                num_nieghbors = neighbors[x][y]

                # every cell can only be a 0 or a 1
                solver.add(Or(target_cell == 1, target_cell == 0))

                # compute allowable living nieghbors
                neighbors_allowed = -target_cell # we don't want to count the current cell as a neighbor 
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        if (x + i <= self.width-1 and x + i >= 0) and \
                            (y + j <= self.height-1 and y + j >= 0):
                            neighbors_allowed += target[x + i][y + j]

                solver.add(neighbors[x][y] == neighbors_allowed)

                # encode the transition rules as SL statements
                if curr_cell == 1:
                    alive_rule = Or(num_nieghbors == 3, And(target_cell == 1, num_nieghbors == 2))
                    solver.add(alive_rule)
                else: 
                    dead_rule = Or(
                        And(target_cell == 0, Not(num_nieghbors == 3)),
                        And(target_cell == 1, Or(num_nieghbors < 2, num_nieghbors > 3))
                    )
                    solver.add(dead_rule)

        
        initial_terms = target.flatten().tolist() + neighbors.flatten().tolist()
        models = list(self.__all_previous(solver, initial_terms))
        candidates = [self.model_to_board(b) for b in models]

        def get_best(candidates):
            best = None
            best_sum = np.Inf
            for c in candidates:
                # print("curr_sum: ",curr_sum)
                curr_sum = np.sum(c)
                if best is None or curr_sum < best_sum:
                    best = c 
                    best_sum = curr_sum

            return best

           
        # add the rejected candidates to the blacklist
        best = get_best(candidates)
        best_id = Board.get_id(best)

        rest = [c for c in candidates if Board.get_id(c) != best_id]
        Board.save_states(rest, self.t-1, debug=False) 

        return best


    def step(self):
        """takes one step forwards"""
        self._board = self.compute_next()
        self.t += 1

    def step_back(self):
        """tries to take one step backwards"""
        prev = self.compute_previous()
        if prev is not None:
            self._board = prev
            self.t -= 1
        else: 
            raise Exception("reached a terminal state")
    
    def play(self, n):
        """takes `n` steps forwards"""
        print(f"taking {n} steps forward from t={self.t}")
        print(self)
        for _ in range(n):
            self.step()
            print(self)

    def rewind(self, n):
        """takes `n` steps backwards"""
        print(f"taking {n} steps backwards from t={self.t}")
        print(self)
        for _ in range(n):
            self.step_back()
            print(self)

    def from_array(arr: np.ndarray):
        w, h = arr.shape
        b = Board(w, h)
        b._board = np.array([row[:] for row in arr], dtype=int)
        return b

    def make_glider(n):
        if n < 3: 
            raise Exception("n too small to make a glider")

        arr = np.zeros(n * n, dtype=int).reshape((n, n))
        arr[1][0] = 1
        arr[2][1] = 1
        arr[0][2] = 1
        arr[1][2] = 1
        arr[2][2] = 1

        result = Board.from_array(arr)

        return result

    def __getitem__(self, x):
        return self._board[x]
    
    def __repr__(self): 
        padding = max(len(str(self.width)), 2)
        result = "".ljust(padding + 1)
        
        for i in range(self.width):
            result += f"{i}".center(padding)

        result += f"\tt: {self.t}\n"

        for x in range(self.width):
            result += f"{x}".rjust(padding) + " "
            for y in range(self.height):
                if self[x][y] == 1: result += "■".center(padding)
                else: result += "▢".center(padding)
            result += "\n"
        result += "\n"
        return result 
    
    def get_id(board: np.ndarray):
        return str(board.flatten().dot(2 ** np.arange(board.size)[::-1]))
    
    def decode_id(id: str, w: int, h: int):
        b_id = int(id)
        leading_zeros = w * h
        bin_str = format(b_id, f"0{leading_zeros}b")
        as_arr = np.array([int(bit) for bit in bin_str]).reshape((w, h))
        return as_arr
    
    # TODO: that's a lot of io
    def save_state(board:np.ndarray, t:int, path="forbidden_states/", debug=True):
        w, h = board.shape
        b_id = Board.get_id(board)
        if debug:
            fname = f"t{t}_({w}x{h})_{b_id}.npy"
            np.savetxt(f"{path}{fname}", board, fmt="%s")
        else: 
            fname = f"t{t}_({w}x{h})_all.txt"
            with open(f"{path}{fname}", "a") as f:
                f.write(f"{b_id}\n")

    def save_states(boards, t:int, path="forbidden_states/", debug=True):
        if debug:
            for board in boards: 
                w, h = board.shape
                b_id = Board.get_id(board)
                fname = f"t{t}_({w}x{h})_{b_id}.npy"
                np.savetxt(f"{path}{fname}", board, fmt="%s")
        else: 
            w, h = boards[0].shape
            fname = f"t{t}_({w}x{h})_all.txt"
            with open(f"{path}{fname}", "a") as f:
                for board in boards: 
                    b_id = Board.get_id(board)
                    f.write(f"{b_id}\n")

    def load_states(self, path="forbidden_states/"):
        w, h = self._board.shape
        boards = [i for i in range(10)]
        for fname in os.listdir(path):
            if f"({w}x{h})" in fname and "_all" in fname:
                t = int(fname[1:2])
                boards[t] = []
                with open(f"{path}{fname}") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip() != "":
                            boards[t].append(Board.decode_id(line, w, h))
                return boards
    
if __name__ == "__main__":
    b = Board.make_glider(4)
    b.play(2)
    b.rewind(1)
    # print(Board.get_id(b._board))
        
