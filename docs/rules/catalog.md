# Rule catalog

A by-topic map of the rules registered in `mm-rules`, organized by ID range and family. This
is a map for finding where a rule lives, not an inventory of what works — it was written by
hand from the rule definitions and has not been individually re-verified against every entry.
**Read [`overview.md`](overview.md) first**: it explains what "registered" does and does not
imply, and gives the commands that measure the real, current numbers.

ID ranges are historical and not fully contiguous; some IDs listed as ranges have gaps.

---

## Algebra (ID 1–15, 300–369)

### Core identities (1–15)
| ID | Name | Formula |
|----|------|---------|
| 1 | `const_fold` | Evaluate constant expressions |
| 2 | `add_identity` | x + 0 = x |
| 3 | `mul_identity` | x × 1 = x |
| 4 | `mul_zero` | x × 0 = 0 |
| 5 | `collect_like` | ax + bx = (a+b)x |
| 6 | `difference_of_squares` | a² - b² = (a+b)(a-b) |
| 7 | `perfect_square_sum` | a² + 2ab + b² = (a+b)² |
| 8 | `perfect_square_diff` | a² - 2ab + b² = (a-b)² |
| 9 | `commutative_add` | a + b = b + a |
| 10 | `commutative_mul` | a × b = b × a |

### Exponents (300–319)
| ID | Name | Formula |
|----|------|---------|
| 300 | `power_add` | xᵃ × xᵇ = xᵃ⁺ᵇ |
| 301 | `power_sub` | xᵃ / xᵇ = xᵃ⁻ᵇ |
| 302 | `power_mul` | (xᵃ)ᵇ = xᵃᵇ |
| 303 | `power_zero` | x⁰ = 1 |
| 304 | `power_one` | x¹ = x |
| 305 | `power_neg` | x⁻ᵃ = 1/xᵃ |
| 306 | `power_frac` | x^(a/b) = ᵇ√(xᵃ) |
| 307 | `power_distribute` | (ab)ⁿ = aⁿbⁿ |
| 308 | `power_quotient` | (a/b)ⁿ = aⁿ/bⁿ |
| 309 | `sqrt_as_power` | √x = x^(1/2) |

### Logarithms and exponentials (320–332)
| ID | Name | Formula |
|----|------|---------|
| 320 | `log_product` | log(ab) = log(a) + log(b) |
| 321 | `log_quotient` | log(a/b) = log(a) - log(b) |
| 322 | `log_power` | log(aⁿ) = n·log(a) |
| 323 | `log_base_change` | logₐ(x) = log(x)/log(a) |
| 324 | `log_one` | log(1) = 0 |
| 325 | `log_same_base` | logₐ(a) = 1 |
| 326 | `exp_product` | eᵃ × eᵇ = eᵃ⁺ᵇ |
| 327 | `exp_quotient` | eᵃ / eᵇ = eᵃ⁻ᵇ |
| 328 | `exp_power` | (eᵃ)ᵇ = eᵃᵇ |
| 329 | `exp_zero` | e⁰ = 1 |
| 330 | `exp_one` | e¹ = e |
| 331 | `exp_ln` | e^(ln x) = x |
| 332 | `ln_exp` | ln(eˣ) = x |

### Radicals (333–344)
| ID | Name | Formula |
|----|------|---------|
| 333 | `sqrt_product` | √(ab) = √a × √b |
| 334 | `sqrt_quotient` | √(a/b) = √a / √b |
| 335 | `sqrt_square` | √(x²) = \|x\| |
| 336 | `cube_root_cube` | ∛(x³) = x |
| 337 | `nth_root_power` | ⁿ√(xⁿ) = x |
| 338 | `rationalize_denom` | 1/√a = √a/a |
| 339 | `conjugate_multiply` | (a+√b)(a-√b) = a²-b |
| 340 | `sum_of_cubes` | a³ + b³ = (a+b)(a² - ab + b²) |
| 341 | `diff_of_cubes` | a³ - b³ = (a-b)(a² + ab + b²) |

### Polynomial theorems (345–354)
Mostly stated as facts rather than executable rewrites (noted where so) — see `overview.md`
on what "informational" means for reachability.
| ID | Name | Formula | |
|----|------|---------|---|
| 345 | `vieta_sum` | r₁ + r₂ = -b/a for ax²+bx+c=0 | informational |
| 346 | `vieta_product` | r₁ · r₂ = c/a for ax²+bx+c=0 | informational |
| 347 | `factor_quadratic` | ax² + bx + c = a(x-r₁)(x-r₂) | informational |
| 348 | `rational_root_test` | rational roots are ±(factors of a₀)/(factors of aₙ) | informational |
| 349 | `synthetic_division` | efficient division by (x-a) | informational |
| 350 | `polynomial_division` | P(x)/Q(x) = S(x) + R(x)/Q(x), deg(R) < deg(Q) | informational |
| 351 | `remainder_theorem` | remainder of P(x)/(x-a) is P(a) | informational |
| 352 | `factor_theorem` | (x-a) divides P(x) ⟺ P(a) = 0 | informational |
| 353 | `bezout_identity` | gcd(a,b) = ax + by for some integers x,y | informational |
| 354 | `euclidean_division` | a = bq + r, 0 ≤ r < b | informational |

### Fraction operations (355–359)
| ID | Name | Formula |
|----|------|---------|
| 355 | `fraction_add` | a/b + c/d = (ad+bc)/bd |
| 356 | `fraction_mul` | (a/b)(c/d) = ac/bd |
| 357 | `fraction_div` | (a/b)/(c/d) = ad/bc |
| 358 | `cross_multiply` | a/b = c/d ⟹ ad = bc |
| 359 | `lcd_combine` | combine fractions via lowest common denominator (informational) |

### Inequalities (360–369)
| ID | Name | Formula |
|----|------|---------|
| 360 | `abs_nonnegative` | \|x\| ≥ 0 |
| 361 | `abs_zero_iff` | \|x\| = 0 ⟺ x = 0 |
| 362 | `triangle_inequality` | \|a + b\| ≤ \|a\| + \|b\| |
| 363 | `reverse_triangle` | \|a - b\| ≥ \|\|a\| - \|b\|\| |
| 364 | `am_gm_2` | (a + b)/2 ≥ √(ab) |
| 365 | `am_gm_3` | (a + b + c)/3 ≥ ∛(abc) |
| 366 | `qm_am` | √((a² + b²)/2) ≥ (a + b)/2 |
| 367 | `cauchy_schwarz_2` | (ab + cd)² ≤ (a² + c²)(b² + d²) |
| 368 | `holders_inequality` | Σaᵢbᵢ ≤ (Σaᵢᵖ)^(1/p)(Σbᵢᵠ)^(1/q) |
| 369 | `minkowski` | (Σ(aᵢ+bᵢ)ᵖ)^(1/p) ≤ (Σaᵢᵖ)^(1/p) + (Σbᵢᵖ)^(1/p) |

> As of the current guardrail (see `mm-search/tests/guardrail_reachability.rs`), several
> inequality rules — those that produce a *bound* rather than an equal value — are
> deliberately kept guardrail-hidden rather than exposed universally, because the verifier
> cannot safely extend rule-replay trust to them near calculus expressions. Check that test's
> doc comment for the current list.

---

## Trigonometry (ID 19–50, 200–269)

### Pythagorean identities (19–23)
| ID | Name | Formula |
|----|------|---------|
| 19 | `pythagorean_sin_cos` | sin²x + cos²x = 1 |
| 20 | `pythagorean_tan_sec` | 1 + tan²x = sec²x |
| 21 | `pythagorean_cot_csc` | 1 + cot²x = csc²x |

### Angle sum/difference (24–29)
| ID | Name | Formula |
|----|------|---------|
| 24 | `sin_sum` | sin(a+b) = sin(a)cos(b) + cos(a)sin(b) |
| 25 | `sin_diff` | sin(a-b) = sin(a)cos(b) - cos(a)sin(b) |
| 26 | `cos_sum` | cos(a+b) = cos(a)cos(b) - sin(a)sin(b) |
| 27 | `cos_diff` | cos(a-b) = cos(a)cos(b) + sin(a)sin(b) |
| 28 | `tan_sum` | tan(a+b) = (tan a + tan b)/(1 - tan a·tan b) |
| 29 | `tan_diff` | tan(a-b) = (tan a - tan b)/(1 + tan a·tan b) |

### Double angle (200–205)
| ID | Name | Formula |
|----|------|---------|
| 200 | `sin_2x` | sin(2x) = 2sin(x)cos(x) |
| 201 | `cos_2x_v1` | cos(2x) = cos²x - sin²x |
| 202 | `cos_2x_v2` | cos(2x) = 2cos²x - 1 |
| 203 | `cos_2x_v3` | cos(2x) = 1 - 2sin²x |
| 204 | `tan_2x` | tan(2x) = 2tan(x)/(1 - tan²x) |

### Hyperbolic functions (220–231)
| ID | Name | Formula |
|----|------|---------|
| 220 | `sinh_def` | sinh(x) = (eˣ - e⁻ˣ)/2 |
| 221 | `cosh_def` | cosh(x) = (eˣ + e⁻ˣ)/2 |
| 222 | `tanh_def` | tanh(x) = sinh(x)/cosh(x) |
| 223 | `sinh_cosh_identity` | cosh²x - sinh²x = 1 |
| 224 | `sinh_add` | sinh(a+b) = sinh(a)cosh(b) + cosh(a)sinh(b) |
| 225 | `cosh_add` | cosh(a+b) = cosh(a)cosh(b) + sinh(a)sinh(b) |
| 226 | `sinh_2x` | sinh(2x) = 2sinh(x)cosh(x) |
| 227 | `cosh_2x` | cosh(2x) = cosh²x + sinh²x |

### Inverse trig (232–236)
| ID | Name | Formula |
|----|------|---------|
| 232 | `arcsin_sin` | arcsin(sin x) = x for x ∈ [-π/2, π/2] |
| 233 | `arccos_cos` | arccos(cos x) = x for x ∈ [0, π] |
| 234 | `arctan_tan` | arctan(tan x) = x for x ∈ (-π/2, π/2) |
| 235 | `sin_arcsin` | sin(arcsin x) = x |
| 236 | `arcsin_arccos` | arcsin(x) + arccos(x) = π/2 |

### Triple/half angle, product-sum (238–259)
| ID | Name | Formula |
|----|------|---------|
| 238 | `sin_3x` | sin(3x) = 3sin(x) - 4sin³(x) |
| 239 | `cos_3x` | cos(3x) = 4cos³(x) - 3cos(x) |
| 240 | `tan_3x` | tan(3x) = (3tan(x) - tan³(x))/(1 - 3tan²(x)) |
| 242 | `sin_half` | sin(x/2) = ±√((1 - cos x)/2) |
| 243 | `cos_half` | cos(x/2) = ±√((1 + cos x)/2) |
| 244 | `tan_half_v1` | tan(x/2) = sin(x)/(1 + cos x) |
| 245 | `tan_half_v2` | tan(x/2) = (1 - cos x)/sin(x) |
| 248 | `cos_cos_prod` | cos(a)cos(b) = ½[cos(a-b) + cos(a+b)] |
| 249 | `sin_sin_prod` | sin(a)sin(b) = ½[cos(a-b) - cos(a+b)] |
| 250 | `sin_cos_prod` | sin(a)cos(b) = ½[sin(a+b) + sin(a-b)] |
| 254 | `sin_sum_to_prod` | sin(a) + sin(b) = 2sin((a+b)/2)cos((a-b)/2) |
| 255 | `sin_diff_to_prod` | sin(a) - sin(b) = 2cos((a+b)/2)sin((a-b)/2) |
| 256 | `cos_sum_to_prod` | cos(a) + cos(b) = 2cos((a+b)/2)cos((a-b)/2) |
| 257 | `cos_diff_to_prod` | cos(a) - cos(b) = -2sin((a+b)/2)sin((a-b)/2) |

### Chebyshev polynomials (266–269)
| ID | Name | Formula |
|----|------|---------|
| 266 | `chebyshev_T0` | T₀(x) = 1 |
| 267 | `chebyshev_T1` | T₁(x) = x |
| 268 | `chebyshev_recurrence` | Tₙ₊₁(x) = 2xTₙ(x) - Tₙ₋₁(x) |
| 269 | `chebyshev_cos` | Tₙ(cos θ) = cos(nθ) |

---

## Calculus (ID 11–18, 51–90, 420–469)

### Basic derivatives (11–18)
| ID | Name | Formula |
|----|------|---------|
| 11 | `power_rule` | d/dx(xⁿ) = n·xⁿ⁻¹ |
| 12 | `constant_rule` | d/dx(c) = 0 |
| 13 | `sum_rule` | d/dx(f + g) = f' + g' |
| 14 | `product_rule` | d/dx(fg) = f'g + fg' |
| 15 | `quotient_rule` | d/dx(f/g) = (f'g - fg')/g² |
| 16 | `chain_rule_sin` | d/dx(sin f) = cos(f)·f' |
| 17 | `chain_rule_cos` | d/dx(cos f) = -sin(f)·f' |
| 18 | `exp_rule` | d/dx(eˣ) = eˣ |

### Advanced derivatives (51–58)
| ID | Name | Formula |
|----|------|---------|
| 51 | `ln_rule` | d/dx(ln x) = 1/x |
| 52 | `tan_derivative` | d/dx(tan x) = sec²x |
| 53 | `sec_derivative` | d/dx(sec x) = sec(x)tan(x) |
| 54 | `csc_derivative` | d/dx(csc x) = -csc(x)cot(x) |
| 55 | `cot_derivative` | d/dx(cot x) = -csc²x |
| 56 | `arcsin_derivative` | d/dx(arcsin x) = 1/√(1-x²) |
| 57 | `arccos_derivative` | d/dx(arccos x) = -1/√(1-x²) |
| 58 | `arctan_derivative` | d/dx(arctan x) = 1/(1+x²) |

### Integration (420–435)
| ID | Name | Formula |
|----|------|---------|
| 420 | `integral_power` | ∫xⁿ dx = xⁿ⁺¹/(n+1) + C |
| 421 | `integral_constant` | ∫k dx = kx + C |
| 422 | `integral_sum` | ∫(f+g) dx = ∫f dx + ∫g dx |
| 423 | `integral_exp` | ∫eˣ dx = eˣ + C |
| 424 | `integral_ln` | ∫(1/x) dx = ln\|x\| + C |
| 425 | `integral_sin` | ∫sin(x) dx = -cos(x) + C |
| 426 | `integral_cos` | ∫cos(x) dx = sin(x) + C |
| 427 | `integral_tan` | ∫tan(x) dx = -ln\|cos x\| + C |
| 428 | `integral_sec2` | ∫sec²(x) dx = tan(x) + C |
| 429 | `integral_csc2` | ∫csc²(x) dx = -cot(x) + C |
| 430 | `integral_sinh` | ∫sinh(x) dx = cosh(x) + C |
| 431 | `integral_cosh` | ∫cosh(x) dx = sinh(x) + C |
| 432 | `integration_by_parts` | ∫u dv = uv - ∫v du |
| 433 | `u_substitution` | ∫f(g(x))g'(x) dx = ∫f(u) du |
| 434 | `partial_fractions` | partial fractions decomposition |
| 435 | `trig_substitution` | trigonometric substitution |

### Limits (436–442)
| ID | Name | Formula |
|----|------|---------|
| 436 | `limit_constant` | lim c = c |
| 437 | `limit_sum` | lim(f+g) = lim f + lim g |
| 438 | `limit_product` | lim(fg) = lim f · lim g |
| 439 | `limit_quotient` | lim(f/g) = lim f / lim g |
| 440 | `limit_power` | lim(fⁿ) = (lim f)ⁿ |
| 441 | `lhopitals_rule` | lim(f/g) = lim(f'/g') for 0/0 or ∞/∞ |
| 442 | `squeeze_theorem` | g ≤ f ≤ h, lim g = lim h = L ⟹ lim f = L |

### Taylor series and power series (443–450)
| ID | Name | Formula |
|----|------|---------|
| 443 | `taylor_exp` | eˣ = Σ xⁿ/n! |
| 444 | `taylor_sin` | sin(x) = Σ (-1)ⁿ x²ⁿ⁺¹/(2n+1)! |
| 445 | `taylor_cos` | cos(x) = Σ (-1)ⁿ x²ⁿ/(2n)! |
| 446 | `taylor_ln` | ln(1+x) = Σ (-1)ⁿ⁺¹ xⁿ/n |
| 447 | `maclaurin_geometric` | 1/(1-x) = Σ xⁿ, \|x\|<1 |
| 448 | `geometric_series` | Σ arⁿ = a/(1-r), \|r\|<1 |
| 449 | `power_series_diff` | d/dx(Σaₙxⁿ) = Σ n·aₙxⁿ⁻¹ |
| 450 | `power_series_int` | ∫(Σaₙxⁿ) dx = Σ aₙxⁿ⁺¹/(n+1) |

### Vector calculus (451–469)
| ID | Name | Formula |
|----|------|---------|
| 451 | `partial_x` | ∂f/∂x |
| 452 | `partial_y` | ∂f/∂y |
| 453 | `partial_z` | ∂f/∂z |
| 454 | `gradient` | ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z) |
| 455 | `divergence` | ∇·F = ∂Fₓ/∂x + ∂Fᵧ/∂y + ∂Fᵤ/∂z |
| 456 | `curl` | ∇×F |
| 457 | `laplacian` | ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z² |
| 462 | `double_integral` | ∬f dA |
| 463 | `triple_integral` | ∭f dV |
| 464 | `line_integral` | ∫_C F·dr |
| 465 | `surface_integral` | ∬_S F·dS |
| 466 | `greens_theorem` | ∮_C P dx + Q dy = ∬_D (∂Q/∂x - ∂P/∂y) dA |
| 467 | `stokes_theorem` | ∮_C F·dr = ∬_S (∇×F)·dS |
| 468 | `divergence_theorem` | ∭_V ∇·F dV = ∬_S F·dS |
| 469 | `jacobian_transform` | dx dy = \|J\| du dv |

---

## Number theory (ID 100–199, 700–744)

### Divisibility and modular arithmetic (100–119)
| ID | Name | Formula |
|----|------|---------|
| 100 | `divides_zero` | n \| 0 for all n |
| 101 | `divides_self` | n \| n |
| 102 | `divides_transitive` | a\|b ∧ b\|c ⟹ a\|c |
| 103 | `divisibility_sum` | a\|b ∧ a\|c ⟹ a\|(b+c) |
| 104 | `divisibility_product` | a\|b ⟹ a\|(bc) |
| 110 | `mod_add` | (a+b) mod n = ((a mod n) + (b mod n)) mod n |
| 111 | `mod_mul` | (ab) mod n = ((a mod n)(b mod n)) mod n |
| 112 | `mod_power` | aᵇ mod n = ((a mod n)ᵇ) mod n |
| 113 | `mod_sub` | (a-b) mod n = ((a mod n) - (b mod n) + n) mod n |

### GCD/LCM (120–126)
| ID | Name | Formula |
|----|------|---------|
| 120 | `gcd_commutative` | gcd(a,b) = gcd(b,a) |
| 121 | `gcd_associative` | gcd(gcd(a,b),c) = gcd(a,gcd(b,c)) |
| 122 | `gcd_with_zero` | gcd(a,0) = a |
| 123 | `gcd_with_one` | gcd(a,1) = 1 |
| 124 | `euclidean_algorithm` | gcd(a,b) = gcd(b, a mod b) |
| 125 | `gcd_lcm_product` | gcd(a,b) × lcm(a,b) = ab |
| 126 | `bezouts_identity` | gcd(a,b) = ax + by for some x,y |

### Famous theorems (700–703)
| ID | Name | Formula |
|----|------|---------|
| 700 | `fermats_little` | aᵖ⁻¹ ≡ 1 (mod p), p prime, gcd(a,p)=1 |
| 701 | `eulers_theorem` | a^φ(n) ≡ 1 (mod n), gcd(a,n)=1 |
| 702 | `wilsons_theorem` | (p-1)! ≡ -1 (mod p), p prime |
| 703 | `chinese_remainder` | a system of congruences has a unique solution mod the product |

### Totient function (720–723)
| ID | Name | Formula |
|----|------|---------|
| 720 | `totient_prime` | φ(p) = p - 1, p prime |
| 721 | `totient_prime_power` | φ(pᵏ) = pᵏ - pᵏ⁻¹ |
| 722 | `totient_multiplicative` | φ(mn) = φ(m)φ(n), gcd(m,n)=1 |
| 723 | `totient_sum` | Σ_{d\|n} φ(d) = n |

### Quadratic residues (730–733)
| ID | Name | Formula |
|----|------|---------|
| 730 | `euler_criterion` | a^((p-1)/2) ≡ (a/p) (mod p) |
| 731 | `legendre_symbol_1` | (a/p) = 1 if a is a QR mod p |
| 732 | `legendre_symbol_neg1` | (a/p) = -1 if a is a non-QR mod p |
| 733 | `quadratic_reciprocity` | (p/q)(q/p) = (-1)^((p-1)(q-1)/4) |

---

## Combinatorics (ID 600–630)

### Binomial coefficients (600–609)
| ID | Name | Formula |
|----|------|---------|
| 600 | `binomial_zero` | C(n,0) = 1 |
| 601 | `binomial_n` | C(n,n) = 1 |
| 602 | `binomial_one` | C(n,1) = n |
| 603 | `binomial_symmetry` | C(n,k) = C(n,n-k) |
| 604 | `pascal_identity` | C(n,k) = C(n-1,k-1) + C(n-1,k) |
| 605 | `binomial_formula` | C(n,k) = n!/(k!(n-k)!) |
| 606 | `binomial_sum_row` | Σ C(n,k) = 2ⁿ |
| 607 | `binomial_alternating` | Σ (-1)ᵏ C(n,k) = 0 |
| 608 | `vandermonde` | C(m+n,r) = Σ C(m,k)C(n,r-k) |
| 609 | `hockey_stick` | Σᵢ₌ᵣⁿ C(i,r) = C(n+1,r+1) |

### Factorials (610–614)
| ID | Name | Formula |
|----|------|---------|
| 610 | `factorial_zero` | 0! = 1 |
| 611 | `factorial_one` | 1! = 1 |
| 612 | `factorial_recurrence` | n! = n × (n-1)! |
| 613 | `permutation` | P(n,k) = n!/(n-k)! |
| 614 | `double_factorial_odd` | (2n-1)!! = (2n)!/(2ⁿn!) |

### Catalan, Fibonacci, and special numbers (620–627)
| ID | Name | Formula |
|----|------|---------|
| 620 | `catalan_formula` | Cₙ = C(2n,n)/(n+1) |
| 621 | `catalan_recurrence` | Cₙ₊₁ = Σ CᵢCₙ₋ᵢ |
| 622 | `fibonacci_recurrence` | Fₙ = Fₙ₋₁ + Fₙ₋₂ |
| 623 | `fibonacci_binet` | Fₙ = (φⁿ - ψⁿ)/√5 |
| 624 | `lucas_identity` | Fₘ₊ₙ = FₘFₙ₊₁ + Fₘ₋₁Fₙ |
| 625 | `stirling_first` | s(n,k), first kind |
| 626 | `stirling_second` | S(n,k), second kind |
| 627 | `inclusion_exclusion` | \|A∪B\| = \|A\| + \|B\| - \|A∩B\| |

---

*This catalog was assembled from two earlier documents (a hand-written topic table and a
per-rule development changelog) and has not been re-verified line by line against the
current rule definitions. If you find an entry that's wrong, fix it in place — don't restore
the changelog format, and don't re-add a "files updated" log; git history already has that.*
