

"""Compute polynomials that vanish on a finite set of planar points.

This script follows the algorithm:
1. List all monomials up to a given total degree.
2. Build the evaluation matrix on the given point set.
3. Compute its null‑space to get vanishing polynomials.
4. Optionally compute a Gröbner basis for the full vanishing ideal.

Run it directly to see the generators degree‑by‑degree.
Adapt the `points` list as needed.

Author: Yudi 
"""

import sympy as sp 

import re

_EXP_RE = re.compile(r"\*\*([0-9]+)")

def to_caret(expr):
    """Convert a SymPy expression to a string with caret notation."""
    return _EXP_RE.sub(r"^\1", str(expr))

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Symbols
x, y = sp.symbols("x y")

# Sample point set E ⊂ ℝ²
points = [(0, 0), (0, 2), (1, 0), (2, 1), (2, 2)]


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def monomials_up_to_degree(deg: int):
    """
    Return the list of monomials x**i * y**j with i + j ≤ deg
    ordered graded‑lex (total degree first, then x‑exponent descending).
    """
    monos = []
    for total in range(deg + 1):
        for i in range(total + 1):
            j = total - i
            monos.append(x ** i * y ** j)
    return monos


def nullspace_polynomials(pts, max_degree=4):
    """
    Yield (degree, [poly₁, …, poly_k]) where the list is a SymPy basis
    of the null‑space of the evaluation matrix for monomials ≤ degree.
    """
    for d in range(max_degree + 1):
        monos = monomials_up_to_degree(d)
        M = sp.Matrix([[m.subs({x: xi, y: yi}) for m in monos] for xi, yi in pts])
        ker = M.nullspace()
        polys = [sp.factor(sum(c * m for c, m in zip(vec, monos))) for vec in ker]
        yield d, polys


def groebner_basis_from_points(pts):
    """
    Compute a reduced Gröbner basis of the vanishing ideal I(pts) using
    all generators up to |pts| total degree.
    """
    gens = []
    for _, polys in nullspace_polynomials(pts, max_degree=len(pts)):
        gens.extend(polys)

    G = sp.groebner(gens, x, y, order="lex")  # x > y
    return list(G)


# ---------------------------------------------------------------------
# Command‑line interface
def main():
    print("Vanishing polynomials degree‑by‑degree:\n")
    for d, polys in nullspace_polynomials(points, max_degree=len(points)):
        if polys:
            # list monomials up to this degree with caret notation
            monos = monomials_up_to_degree(d)
            monos_str = ",".join(to_caret(m) for m in monos)
            print(f"Degree ≤ {d}   monomials = {{{monos_str}}}   nullspace dim = {len(polys)}")
            for p in polys:
                print("   ", to_caret(p))
            print()

    print("-" * 60)
    print("Reduced basis of the vanishing ideal:")
    for g in groebner_basis_from_points(points):
        print("   ",to_caret(g))


if __name__ == "__main__":
    main()