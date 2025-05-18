"""Compute the vanishing ideal of a finite set of planar points.

This module implements an algorithm to find all polynomials that vanish
on a given set of points in ℝ². The procedure is:

1. Define symbols x, y.
2. Generate all monomials x^i * y^j with total degree ≤ max_degree.
3. Filter out monomials divisible by any recorded leading term to obtain
   standard monomials at each degree.
4. Build the evaluation matrix of these monomials at the sample points.
5. Compute its null‑space to extract vanishing polynomials.
6. Record new leading terms and iterate by increasing degree.
7. Collect and print a reduced generating set of the vanishing ideal.

Usage:
    Adjust `points` and `max_degree` as needed. Run the script directly
    to view generators degree‑by‑degree. Optionally, compute a Gröbner basis
    with `sp.groebner(...)` on the collected generators.

Author: Yudi Sun && Long Qian
Encadrant: Jérémy Berthomieu
"""

import sympy as sp 

import re

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_EXP_RE = re.compile(r"\*\*([0-9]+)")

def to_caret(expr):
    """Convert a SymPy expression to a string with caret notation."""
    return _EXP_RE.sub(r"^\1", str(expr))

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Symbols
x, y = sp.symbols("x y")

# # Number of random sample points
# num_points = 100
# # Randomly generate sample point set E ⊂ ℝ²
# points = [(random.random(), random.random()) for _ in range(num_points)]
# Symbols
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

    # Plot points and reduced basis polynomials
    plt.figure()
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    scatter_handle = plt.scatter(xs, ys, color='black')
    handles = [scatter_handle]
    labels = ['Points']
    # Define plotting grid
    x_min, x_max = min(xs) - 1, max(xs) + 1
    y_min, y_max = min(ys) - 1, max(ys) + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    # Draw zero-contour for each generator in the reduced basis
    for idx, p in enumerate(gens, 1):
        f = sp.lambdify((x, y), p, 'numpy')
        zz = f(xx, yy)
        # Plot zero contour with unique color and label
        CS = plt.contour(xx, yy, zz, levels=[0], colors=[f'C{idx}'], linewidths=2)
        # Use a proxy line for the contour in the legend
        handle = Line2D([0], [0], color=f'C{idx}', linewidth=2)
        handles.append(handle)
        labels.append(to_caret(p))
    plt.legend(handles=handles, labels=labels, loc='best')
    plt.title("Reduced basis vanishing curves")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == "__main__":
    main()