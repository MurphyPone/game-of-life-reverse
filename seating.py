from z3 import Bools, Or, Not, And, Solver, PbEq, sat

"""
A = B
J = K
E = F 

G = F, G = E 
H = F, H = E
I = True
D or E 

not (D = B)
not (D and E and I)
(A + B + C + D + E + F + G + H + I + J + K = 6 )
"""

variables = Bools("A B C D E F G H I J K L")
A, B, C, D, E, F, G, H, I, J, K, L = variables
solver = Solver()
solver.add(A == B)
solver.add(J == K)
solver.add(E == F)

solver.add(G == F)
solver.add(G == E)
solver.add(H == F)
solver.add(H == E)

solver.add(I == True)
solver.add(Or(D, E))

solver.add(Not(D == B))
solver.add(Not(And(D, B, I)))

solver.add(PbEq([(v, 1) for v in variables], 6))

def all_smt(s, initial_terms):
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


for i, sol in enumerate(list(all_smt(solver, variables))):
    print("solution ", i)
    print("\ttable 1: ", sorted(list(filter(lambda v: sol[v] == True, sol)), key=lambda c: str(c)))
    print("\ttable 2: ", sorted(list(filter(lambda v: sol[v] == False, sol)), key=lambda c: str(c)))

# print(solver.check(), solver.model())