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
points = [(0, 0), (1, 0),(2,0),(0,1)]


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

def divides(m1, m2):
    """Return True if monomial m2 is divisible by m1."""
    pd1 = m1.as_powers_dict()
    pd2 = m2.as_powers_dict()
    return all(pd2.get(sym, 0) >= exp for sym, exp in pd1.items())

def nullspace_polynomials(pts, max_degree=4):
    """
    Yield (degree, monos, [poly₁, …, poly_k]) where monos is
    the filtered list of standard monomials of degree ≤ d,
    and the list is a SymPy basis of the null‑space of
    the evaluation matrix for those monomials.
    """
    lead_terms = []
    for d in range(max_degree + 1):
        monos = monomials_up_to_degree(d)
        # filter out monomials divisible by any leading term
        monos_filt = [
            m for m in monos
            if not any(hasattr(lt, "as_powers_dict") and divides(lt, m) for lt in lead_terms)
        ]
        # build evaluation matrix
        M = sp.Matrix([[m.subs({x: xi, y: yi}) for m in monos_filt] for xi, yi in pts])
        ker = M.nullspace()
        polys = [sp.factor(sum(c * m for c, m in zip(vec, monos_filt))) for vec in ker]
        # record leading terms for filtering next degrees
        for p in polys:
            poly_obj = sp.Poly(p, x, y)
            exp_tuple = poly_obj.monoms()[0]
            lt = x**exp_tuple[0] * y**exp_tuple[1]
            lead_terms.append(lt)
        yield d, monos_filt, polys




# ---------------------------------------------------------------------
# Command‑line interface
def main():
    print(f"Points E: {points}")
    print("Vanishing polynomials degree‑by‑degree:\n")
    for d, monos, polys in nullspace_polynomials(points, max_degree=len(points)):
        if polys:
            monos_str = ",".join(to_caret(m) for m in monos)
            print(f"Degree ≤ {d}   monomials = {{{monos_str}}}   nullspace dim = {len(polys)}")
            for p in polys:
                print("   ", to_caret(p))
            print()

    # Collect a reduced generating set from nullspace_polynomials
    gens = []
    for _, _, polys in nullspace_polynomials(points, max_degree=len(points)):
        gens.extend(polys)
    print("-" * 60)
    print("Reduced basis of the vanishing ideal:")
    for p in gens:
        print("   ", to_caret(p))


if __name__ == "__main__":
    main()